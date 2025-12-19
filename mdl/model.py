from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .dataset import IO_SEPARATOR_TOKEN_ID, IGNORE_INDEX, MAX_SEQ_LEN, VOCAB_SIZE
from .nn import Embedding, LayerNorm, Linear, TransformerBlock


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    num_examples: int = 1280
    dtype: jnp.dtype = jnp.bfloat16


class Batch(TypedDict):
    input_ids: Int[Array, "B S"]
    attention_mask: Bool[Array, "B S"]
    example_ids: Int[Array, "B"]
    positions_3d: Int[Array, "B S 3"]


Metrics = Dict[str, Float[Array, ""]]


class Model(eqx.Module):
    config: ModelConfig = eqx.field(static=True)

    token_embedding: Embedding
    example_embedding: Embedding
    drop: eqx.nn.Dropout

    blocks: Tuple[TransformerBlock, ...]
    norm: LayerNorm
    lm_head: Linear

    def __init__(self, cfg: ModelConfig, *, key: jax.Array):
        self.config = cfg
        k_tok, k_ex, k_blocks, k_head = jax.random.split(key, 4)

        self.token_embedding = Embedding(cfg.vocab_size, cfg.d_model, key=k_tok)
        self.example_embedding = Embedding(cfg.num_examples, cfg.d_model, key=k_ex)
        self.drop = eqx.nn.Dropout(cfg.dropout)

        keys = jax.random.split(k_blocks, cfg.n_layers)
        self.blocks = tuple(TransformerBlock(cfg, key=kk) for kk in keys)

        self.norm = LayerNorm(cfg.d_model)
        self.lm_head = Linear(
            cfg.d_model, cfg.vocab_size, key=k_head, dtype=cfg.dtype, bias=False
        )

    def __call__(
        self,
        input_ids: Int[Array, "B S"],
        example_ids: Int[Array, "B"],
        *,
        positions_3d: Int[Array, "B S 3"],
        attention_mask: Optional[Bool[Array, "B S"]] = None,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B S V"]:
        B, S = input_ids.shape
        assert (
            S <= self.config.max_seq_len
        ), f"Sequence length {S} exceeds model capacity ({self.config.max_seq_len})."

        inference = inference or (key is None)
        if attention_mask is None:
            attention_mask = jnp.ones((B, S), dtype=jnp.bool_)
        else:
            attention_mask = attention_mask.astype(jnp.bool_)

        pos_xyz = positions_3d.astype(jnp.int32)

        tok = self.token_embedding(input_ids.astype(jnp.int32))
        ex = self.example_embedding(example_ids.astype(jnp.int32))
        x = tok + ex[:, None, :]
        x = x.astype(self.config.dtype)

        drop_key, *layer_keys = (
            (None, *([None] * len(self.blocks)))
            if key is None
            else jax.random.split(key, 1 + len(self.blocks))
        )
        x = self.drop(x, key=drop_key, inference=inference)

        for block, k_layer in zip(self.blocks, layer_keys):
            x = block(
                x,
                attention_mask=attention_mask,
                pos_xyz=pos_xyz,
                key=k_layer,
                inference=inference,
            )
            x = x * attention_mask[..., None]

        x = self.norm(x)
        logits = self.lm_head(x.astype(self.config.dtype))
        return logits


def _cross_entropy_per_token(
    logits: Float[Array, "B T V"], targets: Int[Array, "B T"]
) -> Float[Array, "B T"]:
    logits_f = logits.astype(jnp.float32)
    logp = jax.nn.log_softmax(logits_f, axis=-1)

    safe_t = jnp.clip(targets, 0, logits.shape[-1] - 1).astype(jnp.int32)
    gathered = jnp.take_along_axis(logp, safe_t[..., None], axis=-1)[..., 0]
    nll = -gathered

    valid = targets != IGNORE_INDEX
    return jnp.where(valid, nll, 0.0).astype(jnp.float32)


def compute_loss(
    model: Model,
    batch: Batch,
    *,
    key: Optional[jax.Array] = None,
    inference: bool = False,
) -> Tuple[Float[Array, "B S V"], Metrics]:
    logits = model(
        batch["input_ids"],
        batch["example_ids"],
        attention_mask=batch["attention_mask"],
        positions_3d=batch["positions_3d"],
        key=key,
        inference=inference,
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:].astype(jnp.int32)
    shift_keep = attention_mask[:, 1:]
    shift_targets = jnp.where(shift_keep, shift_targets, jnp.int32(IGNORE_INDEX))

    raw_losses = _cross_entropy_per_token(shift_logits, shift_targets)
    valid_mask = shift_targets != IGNORE_INDEX
    total_valid = valid_mask.sum().astype(jnp.float32)
    loss = raw_losses.sum() / jnp.maximum(total_valid, 1.0)

    preds = jnp.argmax(shift_logits, axis=-1).astype(jnp.int32)
    correct_tokens = (
        jnp.logical_and(valid_mask, preds == shift_targets).sum().astype(jnp.float32)
    )
    correct_tokens = correct_tokens / jnp.maximum(total_valid, 1.0)

    per_example_valid = valid_mask.sum(axis=1)
    per_example_has_valid = per_example_valid > 0
    per_example_all_correct = jnp.all(
        jnp.logical_or(~valid_mask, preds == shift_targets), axis=1
    )
    exact_correct = (
        jnp.logical_and(per_example_has_valid, per_example_all_correct)
        .sum()
        .astype(jnp.float32)
    )
    exact_total = per_example_has_valid.sum().astype(jnp.float32)
    correct_puzzles = exact_correct / jnp.maximum(exact_total, 1.0)

    shift_input_ids = input_ids[:, :-1].astype(jnp.int32)
    is_output_phase = jnp.cumsum(shift_input_ids == IO_SEPARATOR_TOKEN_ID, axis=1) >= 1
    valid_input = valid_mask & ~is_output_phase
    valid_output = valid_mask & is_output_phase

    input_denom = jnp.maximum(valid_input.sum().astype(jnp.float32), 1.0)
    output_denom = jnp.maximum(valid_output.sum().astype(jnp.float32), 1.0)

    input_loss = (raw_losses * valid_input.astype(jnp.float32)).sum() / input_denom
    output_loss = (raw_losses * valid_output.astype(jnp.float32)).sum() / output_denom

    correct_input_tokens = (
        jnp.logical_and(valid_input, preds == shift_targets).sum().astype(jnp.float32)
    )
    correct_output_tokens = (
        jnp.logical_and(valid_output, preds == shift_targets).sum().astype(jnp.float32)
    )
    correct_input_tokens = correct_input_tokens / input_denom
    correct_output_tokens = correct_output_tokens / output_denom

    per_example_input_valid = valid_input.sum(axis=1)
    per_example_input_has_valid = per_example_input_valid > 0
    per_example_input_all_correct = jnp.all(
        jnp.logical_or(~valid_input, preds == shift_targets), axis=1
    )
    input_exact_correct = (
        jnp.logical_and(per_example_input_has_valid, per_example_input_all_correct)
        .sum()
        .astype(jnp.float32)
    )
    input_exact_total = per_example_input_has_valid.sum().astype(jnp.float32)
    correct_input_puzzles = input_exact_correct / jnp.maximum(input_exact_total, 1.0)

    per_example_output_valid = valid_output.sum(axis=1)
    per_example_output_has_valid = per_example_output_valid > 0
    per_example_output_all_correct = jnp.all(
        jnp.logical_or(~valid_output, preds == shift_targets), axis=1
    )
    output_exact_correct = (
        jnp.logical_and(per_example_output_has_valid, per_example_output_all_correct)
        .sum()
        .astype(jnp.float32)
    )
    output_exact_total = per_example_output_has_valid.sum().astype(jnp.float32)
    correct_output_puzzles = output_exact_correct / jnp.maximum(output_exact_total, 1.0)

    metrics = {
        "loss": loss,
        "input_loss": input_loss,
        "output_loss": output_loss,
        "correct_tokens": correct_tokens,
        "correct_puzzles": correct_puzzles,
        "correct_input_tokens": correct_input_tokens,
        "correct_input_puzzles": correct_input_puzzles,
        "correct_output_tokens": correct_output_tokens,
        "correct_output_puzzles": correct_output_puzzles,
    }
    return logits, metrics
