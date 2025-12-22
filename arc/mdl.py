from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from . import dataset as data
from .nn import Embedding, LayerNorm, Linear, Transformer


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = data.VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 10
    dropout: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16


class Model(eqx.Module):
    token_embedding: Embedding
    task_embedding: Embedding
    drop: eqx.nn.Dropout
    blocks: Transformer
    norm: LayerNorm
    lm_head: Linear

    d_model: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        num_tasks: int,
        key: jax.Array,
    ):
        self.d_model = cfg.d_model
        self.dtype = cfg.dtype

        k_tok, k_task, k_blocks, k_head = jax.random.split(key, 4)
        self.token_embedding = Embedding(cfg.vocab_size, cfg.d_model, key=k_tok)
        self.task_embedding = Embedding(num_tasks, cfg.d_model, key=k_task)
        self.drop = eqx.nn.Dropout(cfg.dropout)
        self.blocks = Transformer(
            depth=cfg.n_layers,
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            mlp_dim=cfg.d_ff,
            dropout=cfg.dropout,
            rope_mode="3d",
            rope_max_x=data.MAX_SIZE + 1,
            rope_max_y=data.MAX_SIZE,
            rope_max_z=8,
            is_causal=True,
            rope_skip=0,
            dtype=cfg.dtype,
            key=k_blocks,
        )
        self.norm = LayerNorm(cfg.d_model)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size, key=k_head, dtype=cfg.dtype)

    def __call__(
        self,
        tokens: Int[Array, "B T"],
        task_ids: Int[Array, "B"],
        *,
        positions: Int[Array, "B T 3"],
        attention_mask: Bool[Array, "B T"],
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B T V"]:
        B, _ = tokens.shape
        drop_key, enc_key = (None, None) if key is None else jax.random.split(key)
        inference = inference or key is None

        tok = self.token_embedding(tokens.astype(jnp.int32))
        task = self.task_embedding(task_ids.astype(jnp.int32))
        x = tok + task[:, None, :]
        x = x.astype(self.dtype)

        x = self.drop(x, key=drop_key, inference=inference)
        x = self.blocks(
            x,
            attention_mask=attention_mask,
            positions=positions,
            key=enc_key,
            inference=inference,
        )
        x = self.norm(x)
        logits = self.lm_head(x.astype(self.dtype))
        return logits


Metrics = Dict[str, Float[Array, ""]]


def build_model(cfg, *, num_tasks: int, key: jax.Array) -> Model:
    model_cfg = ModelConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        dtype=getattr(jnp, cfg.dtype),
    )
    return Model(model_cfg, num_tasks=num_tasks, key=key)


def _init_kv_cache(model: Model, batch_size: int, max_len: int):
    caches = []
    for layer in model.blocks.layers:
        attn = layer.attn
        shape = (batch_size, max_len, attn.n_heads, attn.head_dim)
        k_cache = jnp.zeros(shape, dtype=attn.dtype)
        v_cache = jnp.zeros(shape, dtype=attn.dtype)
        caches.append((k_cache, v_cache))
    return tuple(caches)


def _incremental_forward(
    model: Model,
    tokens: Int[Array, "B 1"],
    task_ids: Int[Array, "B"],
    *,
    positions: Int[Array, "B 1 3"],
    caches,
    cache_index: Int[Array, ""],
    key: Optional[jax.Array],
    inference: bool,
) -> Tuple[Float[Array, "B 1 V"], Tuple[Tuple[jax.Array, jax.Array], ...]]:
    drop_key, enc_key = (None, None) if key is None else jax.random.split(key)
    inference = inference or key is None

    tok = model.token_embedding(tokens.astype(jnp.int32))
    task = model.task_embedding(task_ids.astype(jnp.int32))
    x = tok + task[:, None, :]
    x = x.astype(model.dtype)
    x = model.drop(x, key=drop_key, inference=inference)
    x, caches = model.blocks.incremental(
        x,
        positions=positions,
        caches=caches,
        cache_index=cache_index,
        key=enc_key,
        inference=inference,
    )
    x = model.norm(x)
    logits = model.lm_head(x.astype(model.dtype))
    return logits, caches


def _autoregressive_eval(
    model: Model,
    batch: Dict[str, jax.Array],
) -> Tuple[jax.Array, Metrics]:
    tokens = batch["tokens"].astype(jnp.int32)
    positions = batch["positions"].astype(jnp.int32)
    task_ids = batch["task_ids"].astype(jnp.int32)

    batch_size, seq_len = tokens.shape
    prefix_len = data.GRID_SEQ_LEN + 2
    output_len = data.GRID_SEQ_LEN + 1

    prefix_tokens = tokens[:, :prefix_len]
    prefix_positions = positions[:, :prefix_len, :]
    output_targets = tokens[:, prefix_len : prefix_len + output_len]
    output_positions = positions[:, prefix_len : prefix_len + output_len, :]

    caches = _init_kv_cache(model, batch_size, seq_len)
    cache_index = jnp.array(0, dtype=jnp.int32)

    def prefill_step(carry, inputs):
        caches, cache_index, _ = carry
        tok, pos = inputs
        logits, caches = _incremental_forward(
            model,
            tok[:, None],
            task_ids,
            positions=pos[:, None, :],
            caches=caches,
            cache_index=cache_index,
            key=None,
            inference=True,
        )
        cache_index = cache_index + 1
        return (caches, cache_index, logits[:, 0, :]), None

    init_logits = jnp.zeros((batch_size, data.VOCAB_SIZE), dtype=model.dtype)
    prefill_inputs = (
        jnp.swapaxes(prefix_tokens, 0, 1),
        jnp.swapaxes(prefix_positions, 0, 1),
    )
    (caches, cache_index, prev_logits), _ = jax.lax.scan(
        prefill_step, (caches, cache_index, init_logits), prefill_inputs
    )

    def decode_step(carry, inputs):
        (
            caches,
            cache_index,
            prev_logits,
            nll_sum,
            mask_sum,
            correct_sum,
            all_correct,
            has_valid,
        ) = carry
        target, pos = inputs

        logp = jax.nn.log_softmax(prev_logits.astype(jnp.float32), axis=-1)
        safe_target = jnp.clip(target, 0, logp.shape[-1] - 1)
        nll = -jnp.take_along_axis(logp, safe_target[:, None], axis=-1)[:, 0]

        is_cell = target <= data.MAX_COLOR_ID
        nll_sum = nll_sum + jnp.sum(jnp.where(is_cell, nll, 0.0))
        mask_sum = mask_sum + jnp.sum(is_cell.astype(jnp.float32))

        pred = jnp.argmax(prev_logits, axis=-1).astype(jnp.int32)
        correct = (pred == target) & is_cell
        correct_sum = correct_sum + jnp.sum(correct.astype(jnp.float32))
        all_correct = all_correct & jnp.logical_or(~is_cell, pred == target)
        has_valid = has_valid | is_cell

        logits, caches = _incremental_forward(
            model,
            pred[:, None],
            task_ids,
            positions=pos[:, None, :],
            caches=caches,
            cache_index=cache_index,
            key=None,
            inference=True,
        )
        cache_index = cache_index + 1
        return (
            caches,
            cache_index,
            logits[:, 0, :],
            nll_sum,
            mask_sum,
            correct_sum,
            all_correct,
            has_valid,
        ), None

    init_nll = jnp.array(0.0, dtype=jnp.float32)
    init_mask = jnp.array(0.0, dtype=jnp.float32)
    init_correct = jnp.array(0.0, dtype=jnp.float32)
    init_all_correct = jnp.ones((batch_size,), dtype=jnp.bool_)
    init_has_valid = jnp.zeros((batch_size,), dtype=jnp.bool_)
    decode_inputs = (
        jnp.swapaxes(output_targets, 0, 1),
        jnp.swapaxes(output_positions, 0, 1),
    )
    (
        _,
        _,
        _,
        nll_sum,
        mask_sum,
        correct_sum,
        all_correct,
        has_valid,
    ), _ = jax.lax.scan(
        decode_step,
        (
            caches,
            cache_index,
            prev_logits,
            init_nll,
            init_mask,
            init_correct,
            init_all_correct,
            init_has_valid,
        ),
        decode_inputs,
    )

    denom = jnp.maximum(mask_sum, 1.0)
    loss = nll_sum / denom
    pixel_acc = correct_sum / denom
    exact_correct = jnp.sum((all_correct & has_valid).astype(jnp.float32))
    exact_total = jnp.maximum(jnp.sum(has_valid.astype(jnp.float32)), 1.0)
    exact_acc = exact_correct / exact_total

    metrics = {"loss": loss, "pixel_acc": pixel_acc, "exact_acc": exact_acc}
    return loss, metrics


def _masked_mean(values, mask):
    denom = jnp.maximum(mask.sum().astype(jnp.float32), 1.0)
    return (values * mask.astype(jnp.float32)).sum() / denom


def _masked_accuracy(preds, targets, mask):
    correct = (preds == targets) & mask
    denom = jnp.maximum(mask.sum().astype(jnp.float32), 1.0)
    return correct.sum().astype(jnp.float32) / denom


def _masked_exact(preds, targets, mask):
    per_example_valid = mask.sum(axis=1)
    per_example_has_valid = per_example_valid > 0
    per_example_all_correct = jnp.all(
        jnp.logical_or(~mask, preds == targets), axis=1
    )
    exact_correct = jnp.logical_and(
        per_example_has_valid, per_example_all_correct
    ).sum()
    exact_total = per_example_has_valid.sum()
    return exact_correct.astype(jnp.float32) / jnp.maximum(exact_total, 1.0)


def loss_fn(
    model: Model,
    batch: Dict[str, jax.Array],
    *,
    key: Optional[jax.Array],
    inference: bool,
) -> Tuple[jax.Array, Metrics]:
    if inference:
        return _autoregressive_eval(model, batch)

    logits = model(
        batch["tokens"],
        batch["task_ids"],
        attention_mask=batch["attention_mask"],
        positions=batch["positions"],
        key=key,
        inference=inference,
    )

    tokens = batch["tokens"].astype(jnp.int32)
    attention_mask = batch["attention_mask"].astype(jnp.bool_)

    shift_logits = logits[:, :-1, :]
    shift_targets = tokens[:, 1:]
    shift_mask = attention_mask[:, 1:]
    shift_mask = shift_mask & (shift_targets != data.IGNORE_TOKEN_ID)

    logp = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
    safe_targets = jnp.clip(shift_targets, 0, logp.shape[-1] - 1)
    nll = -jnp.take_along_axis(logp, safe_targets[..., None], axis=-1)[..., 0]

    ar_loss = _masked_mean(nll, shift_mask)

    shift_input_ids = tokens[:, :-1]
    output_phase = jnp.cumsum(shift_input_ids == data.IO_SEPARATOR_TOKEN_ID, axis=1) >= 1

    output_cell_mask = (
        shift_mask & output_phase & (shift_targets <= data.MAX_COLOR_ID)
    )
    input_cell_mask = (
        shift_mask & ~output_phase & (shift_targets <= data.MAX_COLOR_ID)
    )

    output_loss = _masked_mean(nll, output_cell_mask)
    input_loss = _masked_mean(nll, input_cell_mask)

    preds = jnp.argmax(shift_logits, axis=-1).astype(jnp.int32)
    output_pixel_acc = _masked_accuracy(preds, shift_targets, output_cell_mask)
    input_pixel_acc = _masked_accuracy(preds, shift_targets, input_cell_mask)
    output_exact_acc = _masked_exact(preds, shift_targets, output_cell_mask)
    input_exact_acc = _masked_exact(preds, shift_targets, input_cell_mask)

    metrics = {
        "loss": output_loss,
        "pixel_acc": output_pixel_acc,
        "exact_acc": output_exact_acc,
        "ar_loss": ar_loss,
        "input_loss": input_loss,
        "output_loss": output_loss,
        "input_pixel_acc": input_pixel_acc,
        "output_pixel_acc": output_pixel_acc,
        "input_exact_acc": input_exact_acc,
        "output_exact_acc": output_exact_acc,
    }
    return ar_loss, metrics
