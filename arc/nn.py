from typing import Optional

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
        k_w, _ = jax.random.split(key)
        param_dtype = jnp.float32
        limit = jnp.sqrt(6.0 / (in_features + out_features))
        self.weight = jax.random.uniform(
            k_w,
            (out_features, in_features),
            minval=-limit,
            maxval=limit,
            dtype=param_dtype,
        )
        self.bias = jnp.zeros((out_features,), dtype=param_dtype) if bias else None

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


def _build_positions(grid_size: int, num_planes: int) -> Int[Array, "T 3"]:
    xs = jnp.arange(grid_size, dtype=jnp.int32)
    ys = jnp.arange(grid_size, dtype=jnp.int32)
    grid_x, grid_y = jnp.meshgrid(xs, ys, indexing="xy")
    base = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    planes = []
    for z in range(num_planes):
        z_col = jnp.full((base.shape[0], 1), z, dtype=jnp.int32)
        planes.append(jnp.concatenate([base, z_col], axis=-1))
    return jnp.concatenate(planes, axis=0)


class RotaryEmbedding3D(eqx.Module):
    cos_x: Float[Array, "MX DX"]
    sin_x: Float[Array, "MX DX"]
    cos_y: Float[Array, "MY DY"]
    sin_y: Float[Array, "MY DY"]
    cos_z: Float[Array, "MZ DZ"]
    sin_z: Float[Array, "MZ DZ"]
    positions: Int[Array, "T 3"]
    head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)
    d_x: int = eqx.field(static=True)
    d_y: int = eqx.field(static=True)
    d_z: int = eqx.field(static=True)
    max_x: int = eqx.field(static=True)
    max_y: int = eqx.field(static=True)
    max_z: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)

    def __init__(
        self,
        head_dim: int,
        *,
        grid_size: int,
        num_planes: int = 2,
        base: float = 10000.0,
    ):
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

        self.max_x = grid_size
        self.max_y = grid_size
        self.max_z = num_planes
        self.seq_len = grid_size * grid_size * num_planes

        self.cos_x, self.sin_x = self._build_cache(self.d_x, self.max_x)
        self.cos_y, self.sin_y = self._build_cache(self.d_y, self.max_y)
        self.cos_z, self.sin_z = self._build_cache(self.d_z, self.max_z)
        self.positions = _build_positions(grid_size, num_planes)

    def _build_cache(self, dim: int, max_pos: int):
        if dim <= 0:
            return (
                jnp.zeros((0, 0), dtype=jnp.float32),
                jnp.zeros((0, 0), dtype=jnp.float32),
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
    ):
        if q.shape[-1] != self.head_dim:
            raise ValueError("q/k last dim must equal head_dim")
        if q.shape[1] > self.seq_len:
            raise ValueError("sequence length exceeds RoPE cache")

        pos = self.positions[: q.shape[1]]
        pos_x = jnp.clip(pos[:, 0], 0, self.max_x - 1)
        pos_y = jnp.clip(pos[:, 1], 0, self.max_y - 1)
        pos_z = jnp.clip(pos[:, 2], 0, self.max_z - 1)

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

        cos = lax.stop_gradient(cos).astype(jnp.float32)[None, :, None, :]
        sin = lax.stop_gradient(sin).astype(jnp.float32)[None, :, None, :]

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        q_out = qf * cos + _rotate_half(qf) * sin
        k_out = kf * cos + _rotate_half(kf) * sin
        return q_out.astype(q.dtype), k_out.astype(k.dtype)


class SelfAttention(eqx.Module):
    qkv_proj: Linear
    out_proj: Linear
    attn_dropout: eqx.nn.Dropout
    proj_dropout: eqx.nn.Dropout
    rope: RotaryEmbedding3D
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        *,
        rope: RotaryEmbedding3D,
        dtype: jnp.dtype,
        key: jax.Array,
    ):
        self.n_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dtype = dtype
        self.rope = rope

        k_qkv, k_out = jax.random.split(key, 2)
        self.qkv_proj = Linear(
            embed_dim, 3 * embed_dim, key=k_qkv, dtype=dtype, bias=False
        )
        self.out_proj = Linear(embed_dim, embed_dim, key=k_out, dtype=dtype, bias=False)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.proj_dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask: Optional[Bool[Array, "B T"]],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        inference = inference or (key is None)
        k_attn, k_proj = (None, None) if key is None else jax.random.split(key, 2)

        x = x.astype(self.dtype)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q, k = self.rope(q, k)
        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        attn_out = _flash_attention(q, k, v, attention_mask=attention_mask)
        attn_out = self.attn_dropout(attn_out, key=k_attn, inference=inference)
        attn_out = attn_out.reshape(x.shape[0], x.shape[1], -1)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out, key=k_proj, inference=inference)
        return attn_out.astype(self.dtype)


class FeedForward(eqx.Module):
    fc1: Linear
    fc2: Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float,
        *,
        dtype: jnp.dtype,
        key,
    ):
        k1, k2 = jax.random.split(key, 2)
        self.fc1 = Linear(embed_dim, mlp_dim, key=k1, dtype=dtype)
        self.fc2 = Linear(mlp_dim, embed_dim, key=k2, dtype=dtype)
        self.drop1 = eqx.nn.Dropout(dropout)
        self.drop2 = eqx.nn.Dropout(dropout)
        self.dtype = dtype

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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        *,
        rope: RotaryEmbedding3D,
        dtype: jnp.dtype,
        key: jax.Array,
    ):
        k_attn, k_ff = jax.random.split(key, 2)
        self.ln1 = LayerNorm(embed_dim)
        self.attn = SelfAttention(
            embed_dim,
            num_heads,
            dropout,
            rope=rope,
            dtype=dtype,
            key=k_attn,
        )
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, mlp_dim, dropout, dtype=dtype, key=k_ff)
        self.dtype = dtype

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask: Optional[Bool[Array, "B T"]],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        k_attn, k_ff = (None, None) if key is None else jax.random.split(key, 2)
        inference = inference or (key is None)

        h = self.ln1(x)
        a = self.attn(
            h,
            attention_mask=attention_mask,
            key=k_attn,
            inference=inference,
        )
        x = x + a

        h2 = self.ln2(x)
        f = self.ff(h2, key=k_ff, inference=inference)
        x = x + f
        return x.astype(self.dtype)


class Transformer(eqx.Module):
    layers: tuple

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        *,
        grid_size: int,
        num_planes: int,
        dtype: jnp.dtype,
        key: jax.Array,
    ):
        rope = RotaryEmbedding3D(
            head_dim=embed_dim // num_heads,
            grid_size=grid_size,
            num_planes=num_planes,
        )
        keys = jax.random.split(key, depth)
        self.layers = tuple(
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_dim,
                dropout,
                rope=rope,
                dtype=dtype,
                key=layer_key,
            )
            for layer_key in keys
        )

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask: Optional[Bool[Array, "B T"]],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        layer_keys = (
            [None] * len(self.layers)
            if key is None
            else list(jax.random.split(key, len(self.layers)))
        )
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(
                x,
                attention_mask=attention_mask,
                key=layer_key,
                inference=inference,
            )
        return x


def _flash_attention(
    q: Float[Array, "B T H D"],
    k: Float[Array, "B T H D"],
    v: Float[Array, "B T H D"],
    *,
    attention_mask: Optional[Bool[Array, "B T"]],
) -> Float[Array, "B T H D"]:
    mask = None
    if attention_mask is not None:
        mask = attention_mask.astype(jnp.bool_)[:, None, None, :]
    return jax.nn.dot_product_attention(
        query=q,
        key=k,
        value=v,
        mask=mask,
        bias=None,
        is_causal=False,
    )


def _rotate_half(x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    orig = x.shape
    x = x.reshape(orig[:-1] + (-1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    y = jnp.stack([-x2, x1], axis=-1)
    return y.reshape(orig)
