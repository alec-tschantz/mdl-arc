from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from . import dataset as data
from .nn import PatchEmbed, LayerNorm, Linear, Transformer


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = data.VOCAB_SIZE
    grid_size: int = 32
    patch_size: int = 4
    num_support: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 10
    dropout: float = 0.1
    pad_loss_weight: float = 0.0
    loss_on_query_only: bool = False
    dtype: jnp.dtype = jnp.bfloat16


class Model(eqx.Module):
    patch_embed: PatchEmbed
    drop: eqx.nn.Dropout
    blocks: Transformer
    norm: LayerNorm
    head: Linear

    grid_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_support: int = eqx.field(static=True)
    patch_dim: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)
    pad_loss_weight: float = eqx.field(static=True)
    loss_on_query_only: bool = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        key: jax.Array,
    ):
        if cfg.grid_size % cfg.patch_size != 0:
            raise ValueError("grid_size must be divisible by patch_size")

        self.grid_size = cfg.grid_size
        self.patch_size = cfg.patch_size
        self.num_support = cfg.num_support
        self.patch_dim = cfg.patch_size * cfg.patch_size
        self.vocab_size = cfg.vocab_size
        self.pad_loss_weight = cfg.pad_loss_weight
        self.loss_on_query_only = cfg.loss_on_query_only
        self.dtype = cfg.dtype

        patch_grid = cfg.grid_size // cfg.patch_size

        k_patch, k_blocks, k_head = jax.random.split(key, 3)
        self.patch_embed = PatchEmbed(
            cfg.grid_size,
            cfg.patch_size,
            cfg.vocab_size,
            cfg.d_model,
            key=k_patch,
            dtype=cfg.dtype,
        )
        self.drop = eqx.nn.Dropout(cfg.dropout)
        self.blocks = Transformer(
            depth=cfg.n_layers,
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            mlp_dim=cfg.d_ff,
            dropout=cfg.dropout,
            rope_mode="4d",
            rope_max_io=2,
            rope_max_x=patch_grid,
            rope_max_y=patch_grid,
            rope_max_example=cfg.num_support + 1,
            is_causal=True,
            rope_skip=0,
            dtype=cfg.dtype,
            key=k_blocks,
        )
        self.norm = LayerNorm(cfg.d_model)
        self.head = Linear(
            cfg.d_model, cfg.vocab_size * self.patch_dim, key=k_head, dtype=cfg.dtype
        )

    def _embed_grids(self, grids: Int[Array, "B G H W"]) -> Float[Array, "B T D"]:
        bsz, num_grids, height, width = grids.shape
        flat = grids.reshape(bsz * num_grids, height, width)
        one_hot = jax.nn.one_hot(flat.astype(jnp.int32), self.vocab_size)
        one_hot = jnp.transpose(one_hot, (0, 3, 1, 2))
        patch_tokens = jax.vmap(self.patch_embed)(one_hot)
        tokens_per_grid = self.patch_embed.grid * self.patch_embed.grid
        return patch_tokens.reshape(bsz, num_grids * tokens_per_grid, -1)

    def __call__(
        self,
        grids: Int[Array, "B G H W"],
        *,
        positions: Int[Array, "B T 4"],
        attention_mask: Bool[Array, "B T"],
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B T P V"]:
        drop_key, enc_key = (None, None) if key is None else jax.random.split(key)
        inference = inference or key is None

        x = self._embed_grids(grids)
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
        logits = self.head(x.astype(self.dtype))
        logits = logits.reshape(x.shape[0], x.shape[1], self.patch_dim, self.vocab_size)
        return logits


Metrics = Dict[str, Float[Array, ""]]


def _patchify_grids(
    grids: Int[Array, "B G H W"], patch_size: int
) -> Int[Array, "B T P"]:
    bsz, num_grids, height, width = grids.shape
    grid = grids.reshape(
        bsz,
        num_grids,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    grid = jnp.transpose(grid, (0, 1, 2, 4, 3, 5))
    patches = grid.reshape(
        bsz,
        num_grids,
        (height // patch_size) * (width // patch_size),
        patch_size * patch_size,
    )
    return patches.reshape(bsz, -1, patch_size * patch_size)


def _weighted_mean(values, weights):
    denom = jnp.maximum(weights.sum().astype(jnp.float32), 1.0)
    return (values * weights.astype(jnp.float32)).sum() / denom


def _masked_accuracy(preds, targets, mask):
    correct = (preds == targets) & mask
    denom = jnp.maximum(mask.sum().astype(jnp.float32), 1.0)
    return correct.sum().astype(jnp.float32) / denom


def _masked_exact(preds, targets, mask):
    per_example_valid = mask.sum(axis=(1, 2))
    per_example_has_valid = per_example_valid > 0
    per_example_all_correct = jnp.all(
        jnp.logical_or(~mask, preds == targets), axis=(1, 2)
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
    logits = model(
        batch["grids"],
        attention_mask=batch["attention_mask"],
        positions=batch["positions"],
        key=key,
        inference=inference,
    )

    grids = batch["grids"].astype(jnp.int32)
    patches = _patchify_grids(grids, model.patch_size)

    shift_logits = logits[:, :-1, :, :]
    shift_targets = patches[:, 1:, :]

    logp = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
    safe_targets = jnp.clip(shift_targets, 0, model.vocab_size - 1)
    nll = -jnp.take_along_axis(logp, safe_targets[..., None], axis=-1)[..., 0]

    cell_mask = shift_targets != data.IGNORE_TOKEN_ID
    preds = jnp.argmax(shift_logits, axis=-1).astype(jnp.int32)
    accuracy = _masked_accuracy(preds, shift_targets, cell_mask)

    tokens_per_grid = model.patch_embed.grid * model.patch_embed.grid
    query_grid_index = 2 * model.num_support + 1
    query_start = query_grid_index * tokens_per_grid
    query_end = query_start + tokens_per_grid
    token_idx = jnp.arange(shift_targets.shape[1]) + 1
    query_token_mask = (token_idx >= query_start) & (token_idx < query_end)
    query_mask = cell_mask & query_token_mask[None, :, None]
    query_token_mask = query_token_mask.astype(jnp.float32)[None, :, None]
    query_weights = jnp.where(cell_mask, 1.0, model.pad_loss_weight) * query_token_mask
    query_loss = _weighted_mean(nll, query_weights)
    query_accuracy = _masked_accuracy(preds, shift_targets, query_mask)
    exact_accuracy = _masked_exact(preds, shift_targets, query_mask)

    if model.loss_on_query_only:
        total_loss = query_loss
    else:
        total_weights = jnp.where(cell_mask, 1.0, model.pad_loss_weight)
        total_loss = _weighted_mean(nll, total_weights)

    metrics = {
        "loss": total_loss,
        "query_loss": query_loss,
        "accuracy": accuracy,
        "query_accuracy": query_accuracy,
        "exact_accuracy": exact_accuracy,
    }
    return total_loss, metrics
