from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, nn as jnn
from jaxtyping import Array, Bool, Float, Int


class Linear(eqx.Module):
    weight: Float[Array, "D_OUT D_IN"]
    bias: Optional[Float[Array, "D_OUT"]]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.use_bias = bias

        k_w, k_b = jax.random.split(key)
        param_dtype = jnp.float32

        limit = jnp.sqrt(6.0 / (in_features + out_features))
        self.weight = jax.random.uniform(
            k_w,
            (out_features, in_features),
            minval=-limit,
            maxval=limit,
            dtype=param_dtype,
        )
        if bias:
            self.bias = jnp.zeros((out_features,), dtype=param_dtype)
        else:
            self.bias = None

    def __call__(self, x: Float[Array, "... D_IN"]) -> Float[Array, "... D_OUT"]:
        x_dtype = x.dtype
        x_low = x.astype(self.dtype)
        w_low = self.weight.astype(self.dtype)
        y = lax.dot_general(
            x_low,
            w_low.T,
            dimension_numbers=(((x.ndim - 1,), (0,)), ((), (()))),
        )
        if self.bias is not None:
            y = y.astype(jnp.float32) + self.bias
        return y.astype(x_dtype)


class LayerNorm(eqx.Module):
    weight: Float[Array, "D"]
    bias: Float[Array, "D"]
    normalized_shape: int = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-5)

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = jnp.ones((normalized_shape,), dtype=jnp.float32)
        self.bias = jnp.zeros((normalized_shape,), dtype=jnp.float32)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x_dtype = x.dtype
        xf = x.astype(jnp.float32)
        mean = xf.mean(axis=-1, keepdims=True)
        var = jnp.mean((xf - mean) ** 2, axis=-1, keepdims=True)
        xhat = (xf - mean) * lax.rsqrt(var + self.eps)
        y = xhat * self.weight + self.bias
        return y.astype(x_dtype)


class Embedding(eqx.Module):
    weight: Float[Array, "N D"]
    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(self, num_embeddings: int, embedding_dim: int, *, key: jax.Array):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = 0.02 * jax.random.normal(
            key, (num_embeddings, embedding_dim), dtype=jnp.float32
        )

    def __call__(self, ids: Int[Array, "..."]) -> Float[Array, "... D"]:
        return jnp.take(self.weight, ids, axis=0)


class RotaryEmbedding3D(eqx.Module):
    cos_x: Float[Array, "MX DX"]
    sin_x: Float[Array, "MX DX"]
    cos_y: Float[Array, "MY DY"]
    sin_y: Float[Array, "MY DY"]
    cos_z: Float[Array, "MZ DZ"]
    sin_z: Float[Array, "MZ DZ"]

    head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)
    d_x: int = eqx.field(static=True)
    d_y: int = eqx.field(static=True)
    d_z: int = eqx.field(static=True)
    max_x: int = eqx.field(static=True)
    max_y: int = eqx.field(static=True)
    max_z: int = eqx.field(static=True)

    def __init__(self, head_dim: int, *, base: float = 10000.0):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        self.base = base

        n_pairs = head_dim // 2
        px = n_pairs // 3
        py = n_pairs // 3
        pz = n_pairs - px - py
        self.d_x = px * 2
        self.d_y = py * 2
        self.d_z = pz * 2

        self.max_x = 32
        self.max_y = 32
        self.max_z = 8

        self.cos_x, self.sin_x = self._build_cache(self.d_x, self.max_x)
        self.cos_y, self.sin_y = self._build_cache(self.d_y, self.max_y)
        self.cos_z, self.sin_z = self._build_cache(self.d_z, self.max_z)

    def _build_cache(
        self, dim: int, max_pos: int
    ) -> Tuple[Float[Array, "M D"], Float[Array, "M D"]]:
        if dim <= 0:
            return jnp.zeros((0, 0), dtype=jnp.float32), jnp.zeros(
                (0, 0), dtype=jnp.float32
            )
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        pos = jnp.arange(max_pos, dtype=jnp.float32)
        t = pos[:, None] * inv_freq[None, :]
        cos = jnp.cos(t)
        sin = jnp.sin(t)
        cos = jnp.repeat(cos, 2, axis=-1)
        sin = jnp.repeat(sin, 2, axis=-1)
        return cos, sin

    def __call__(
        self,
        q: Float[Array, "B T H D"],
        k: Float[Array, "B T H D"],
        pos_xyz: Int[Array, "B T 3"],
    ) -> Tuple[Float[Array, "B T H D"], Float[Array, "B T H D"]]:
        if q.shape[-1] != self.head_dim:
            raise ValueError("q/k last dim must equal head_dim")
        pos_x = jnp.clip(pos_xyz[..., 0], 0, self.max_x - 1)
        pos_y = jnp.clip(pos_xyz[..., 1], 0, self.max_y - 1)
        pos_z = jnp.clip(pos_xyz[..., 2], 0, self.max_z - 1)

        parts_cos = []
        parts_sin = []
        if self.d_x > 0:
            parts_cos.append(self.cos_x[pos_x])
            parts_sin.append(self.sin_x[pos_x])
        if self.d_y > 0:
            parts_cos.append(self.cos_y[pos_y])
            parts_sin.append(self.sin_y[pos_y])
        if self.d_z > 0:
            parts_cos.append(self.cos_z[pos_z])
            parts_sin.append(self.sin_z[pos_z])

        cos = (
            jnp.concatenate(parts_cos, axis=-1)
            if parts_cos
            else jnp.zeros(pos_x.shape + (0,), dtype=jnp.float32)
        )
        sin = (
            jnp.concatenate(parts_sin, axis=-1)
            if parts_sin
            else jnp.zeros(pos_x.shape + (0,), dtype=jnp.float32)
        )

        cos = lax.stop_gradient(cos).astype(jnp.float32)[:, :, None, :]
        sin = lax.stop_gradient(sin).astype(jnp.float32)[:, :, None, :]

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        q_out = qf * cos + _rotate_half(qf) * sin
        k_out = kf * cos + _rotate_half(kf) * sin
        return q_out.astype(q.dtype), k_out.astype(k.dtype)


class SelfAttention(eqx.Module):
    qkv_proj: Linear
    out_proj: Linear
    rope: RotaryEmbedding3D
    dropout: eqx.nn.Dropout

    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, cfg, *, key: jax.Array):
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dtype = cfg.dtype
        self.dropout_rate = float(cfg.dropout)

        k_qkv, k_out = jax.random.split(key, 2)
        self.qkv_proj = Linear(
            cfg.d_model, 3 * cfg.d_model, key=k_qkv, dtype=cfg.dtype, bias=False
        )
        self.out_proj = Linear(
            cfg.d_model, cfg.d_model, key=k_out, dtype=cfg.dtype, bias=False
        )
        self.dropout = eqx.nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding3D(self.head_dim)

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask: Optional[Bool[Array, "B T"]],
        pos_xyz: Int[Array, "B T 3"],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        B, S, E = x.shape
        inference = inference or (key is None)
        drop_key = None if key is None else key

        x = x.astype(self.dtype)

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q, k = self.rope(q, k, pos_xyz)

        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        attn_out = _flash_attention(q, k, v, attention_mask=attention_mask)
        attn_out = self.dropout(attn_out, key=drop_key, inference=inference)

        attn_out = attn_out.reshape(B, S, E)
        attn_out = self.out_proj(attn_out)
        return attn_out.astype(self.dtype)


class FeedForward(eqx.Module):
    fc1: Linear
    fc2: Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, cfg, *, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.fc1 = Linear(cfg.d_model, cfg.d_ff, key=k1, dtype=cfg.dtype)
        self.fc2 = Linear(cfg.d_ff, cfg.d_model, key=k2, dtype=cfg.dtype)
        self.drop1 = eqx.nn.Dropout(cfg.dropout)
        self.drop2 = eqx.nn.Dropout(cfg.dropout)
        self.dtype = cfg.dtype

    def __call__(
        self, x: Float[Array, "B T D"], *, key: Optional[jax.Array], inference: bool
    ) -> Float[Array, "B T D"]:
        inference = inference or (key is None)
        k_mid, k_out = (None, None) if key is None else jax.random.split(key, 2)

        x = x.astype(self.dtype)
        h = self.fc1(x)
        h = jnn.gelu(h)
        h = self.drop1(h, key=k_mid, inference=inference)
        h = self.fc2(h)
        h = self.drop2(h, key=k_out, inference=inference)
        return h.astype(self.dtype)


class TransformerBlock(eqx.Module):
    ln1: LayerNorm
    attn: SelfAttention
    ln2: LayerNorm
    ff: FeedForward
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, cfg, *, key: jax.Array):
        k_attn, k_ff = jax.random.split(key, 2)
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg, key=k_attn)
        self.ln2 = LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg, key=k_ff)
        self.dtype = cfg.dtype

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask: Optional[Bool[Array, "B T"]],
        pos_xyz: Int[Array, "B T 3"],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        k_attn, k_ff = (None, None) if key is None else jax.random.split(key, 2)
        inference = inference or (key is None)

        x = x.astype(self.dtype)
        h = self.ln1(x)
        a = self.attn(
            h,
            attention_mask=attention_mask,
            pos_xyz=pos_xyz,
            key=k_attn,
            inference=inference,
        )
        x = x + a

        h2 = self.ln2(x)
        f = self.ff(h2, key=k_ff, inference=inference)
        x = x + f
        return x.astype(self.dtype)


def _flash_attention(
    q: Float[Array, "B T H D"],
    k: Float[Array, "B T H D"],
    v: Float[Array, "B T H D"],
    *,
    attention_mask: Optional[Bool[Array, "B T"]],
) -> Float[Array, "B T H D"]:
    B, q_len, H, D = q.shape
    s_len = k.shape[1]

    def _pad(x: Float[Array, "B T H D"], pad_len: int) -> Float[Array, "B T H D"]:
        if pad_len == 0:
            return x
        return jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    padded_q_len = ((q_len + 3) // 4) * 4
    padded_s_len = ((s_len + 3) // 4) * 4
    pad_q = padded_q_len - q_len
    pad_s = padded_s_len - s_len

    q_p = _pad(q, pad_q)
    k_p = _pad(k, pad_s)
    v_p = _pad(v, pad_s)

    if attention_mask is None:
        seq_lens = jnp.full((B,), q_len, dtype=jnp.int32)
    else:
        seq_lens = attention_mask.astype(jnp.int32).sum(axis=1)

    out = jax.nn.dot_product_attention(
        query=q_p,
        key=k_p,
        value=v_p,
        mask=None,
        bias=None,
        implementation="cudnn",
        is_causal=True,
        query_seq_lengths=seq_lens,
        key_value_seq_lengths=seq_lens,
    )
    return out[:, :q_len, :, :]


def _rotate_half(x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    orig = x.shape
    x = x.reshape(orig[:-1] + (-1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    y = jnp.stack([-x2, x1], axis=-1)
    return y.reshape(orig)
