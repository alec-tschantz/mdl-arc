from dataclasses import dataclass
from typing import Optional

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
