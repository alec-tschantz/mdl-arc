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
            grid_size=data.MAX_SIZE,
            num_planes=2,
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
        attention_mask: Bool[Array, "B T"],
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B T V"]:
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


def _flatten_grid(x: jax.Array) -> jax.Array:
    return x.reshape(x.shape[0], -1)


def _loss_and_metrics(
    logits: jax.Array,
    targets: jax.Array,
) -> Tuple[jax.Array, Metrics]:
    targets_flat = _flatten_grid(targets.astype(jnp.int32))
    output_logits = logits[:, data.GRID_LEN :, :]

    logp = jax.nn.log_softmax(output_logits.astype(jnp.float32), axis=-1)
    safe_targets = jnp.clip(targets_flat, 0, logp.shape[-1] - 1)
    nll = -jnp.take_along_axis(logp, safe_targets[..., None], axis=-1)[..., 0]

    pad_mask = targets_flat == data.PAD_TOKEN_ID
    nonpad_mask = ~pad_mask
    nonpad_count = jnp.maximum(nonpad_mask.sum(), 1.0)

    total_loss = nll.mean()
    nonpad_loss = (nll * nonpad_mask).sum() / nonpad_count

    preds = jnp.argmax(output_logits, axis=-1).astype(jnp.int32)
    total_pixel_acc = (preds == targets_flat).mean()
    nonpad_pixel_acc = ((preds == targets_flat) & nonpad_mask).sum() / nonpad_count

    total_exact_acc = jnp.mean(jnp.all(preds == targets_flat, axis=1))
    per_example_valid = nonpad_mask.sum(axis=1)
    per_example_has_valid = per_example_valid > 0
    per_example_all_correct = jnp.all(
        jnp.logical_or(~nonpad_mask, preds == targets_flat), axis=1
    )
    nonpad_exact_acc = (
        jnp.sum(per_example_all_correct & per_example_has_valid)
        / jnp.maximum(jnp.sum(per_example_has_valid), 1.0)
    )

    metrics = {
        "loss": total_loss,
        "pixel_acc": total_pixel_acc,
        "exact_acc": total_exact_acc,
        "nonpad_loss": nonpad_loss,
        "nonpad_pixel_acc": nonpad_pixel_acc,
        "nonpad_exact_acc": nonpad_exact_acc,
    }
    return total_loss, metrics


def loss_fn(
    model: Model,
    batch: Dict[str, jax.Array],
    *,
    key: Optional[jax.Array],
    inference: bool,
) -> Tuple[jax.Array, Metrics]:
    inputs = batch["inputs"]
    targets = batch["targets"]
    task_ids = batch["task_ids"]
    attention_mask = batch["attention_mask"]

    batch_size = inputs.shape[0]
    inputs_flat = _flatten_grid(inputs)
    output_queries = jnp.full(
        (batch_size, data.GRID_LEN), data.MASK_TOKEN_ID, dtype=jnp.int32
    )
    tokens = jnp.concatenate([inputs_flat, output_queries], axis=1)

    logits = model(
        tokens,
        task_ids,
        attention_mask=attention_mask,
        key=key,
        inference=inference,
    )
    return _loss_and_metrics(logits, targets)
