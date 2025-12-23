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


class PatchEmbed(eqx.Module):
    conv: eqx.nn.Conv2d
    grid: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.grid = image_size // patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dtype = dtype
        param_dtype = jnp.float32
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=True,
            padding=0,
            dtype=param_dtype,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(self.conv.weight.dtype)
        x = self.conv(x)
        x = jnp.transpose(x, (1, 2, 0))
        x = x.reshape(self.grid * self.grid, self.embed_dim)
        return x.astype(self.dtype)


class LocalMixer(eqx.Module):
    weight: Float[Array, "K K 1 C"]
    bias: Optional[Float[Array, "C"]]
    kernel_size: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        key: jax.Array,
        dtype: jnp.dtype = jnp.bfloat16,
        bias: bool = True,
    ):
        self.kernel_size = kernel_size
        self.dtype = dtype
        k_w, k_b = jax.random.split(key)
        scale = 0.02
        self.weight = scale * jax.random.normal(
            k_w,
            (kernel_size, kernel_size, 1, channels),
            dtype=jnp.float32,
        )
        if bias:
            self.bias = jnp.zeros((channels,), dtype=jnp.float32)
        else:
            self.bias = None

    def __call__(
        self, x: Float[Array, "N H W C"], *, mask: Optional[Bool[Array, "N H W"]]
    ) -> Float[Array, "N H W C"]:
        x_dtype = x.dtype
        xf = x.astype(jnp.float32)
        if mask is not None:
            mask_f = mask.astype(xf.dtype)
            xf = xf * mask_f[..., None]
        y = lax.conv_general_dilated(
            xf,
            self.weight,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=xf.shape[-1],
        )
        if self.bias is not None:
            y = y + self.bias
        if mask is not None:
            y = y * mask_f[..., None]
        return y.astype(x_dtype)


class RotaryEmbedding4D(eqx.Module):
    cos_io: Float[Array, "MI DIO"]
    sin_io: Float[Array, "MI DIO"]
    cos_x: Float[Array, "MX DX"]
    sin_x: Float[Array, "MX DX"]
    cos_y: Float[Array, "MY DY"]
    sin_y: Float[Array, "MY DY"]
    cos_example: Float[Array, "ME DE"]
    sin_example: Float[Array, "ME DE"]
    head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)
    d_io: int = eqx.field(static=True)
    d_x: int = eqx.field(static=True)
    d_y: int = eqx.field(static=True)
    d_example: int = eqx.field(static=True)
    max_io: int = eqx.field(static=True)
    max_x: int = eqx.field(static=True)
    max_y: int = eqx.field(static=True)
    max_example: int = eqx.field(static=True)

    def __init__(
        self,
        head_dim: int,
        *,
        max_io: int,
        max_x: int,
        max_y: int,
        max_example: int,
        base: float = 10000.0,
    ):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        self.base = base

        n_pairs = head_dim // 2
        p_io = n_pairs // 4
        p_x = n_pairs // 4
        p_y = n_pairs // 4
        p_example = n_pairs - p_io - p_x - p_y
        self.d_io = p_io * 2
        self.d_x = p_x * 2
        self.d_y = p_y * 2
        self.d_example = p_example * 2

        self.max_io = max_io
        self.max_x = max_x
        self.max_y = max_y
        self.max_example = max_example

        self.cos_io, self.sin_io = self._build_cache(self.d_io, self.max_io)
        self.cos_x, self.sin_x = self._build_cache(self.d_x, self.max_x)
        self.cos_y, self.sin_y = self._build_cache(self.d_y, self.max_y)
        self.cos_example, self.sin_example = self._build_cache(
            self.d_example, self.max_example
        )

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
        pos_ioxy: Int[Array, "B T 4"],
    ):
        if q.shape[-1] != self.head_dim:
            raise ValueError("q/k last dim must equal head_dim")
        pos_io = jnp.clip(pos_ioxy[..., 0], 0, self.max_io - 1)
        pos_x = jnp.clip(pos_ioxy[..., 1], 0, self.max_x - 1)
        pos_y = jnp.clip(pos_ioxy[..., 2], 0, self.max_y - 1)
        pos_example = jnp.clip(pos_ioxy[..., 3], 0, self.max_example - 1)

        parts_cos = []
        parts_sin = []
        if self.d_io > 0:
            parts_cos.append(self.cos_io[pos_io])
            parts_sin.append(self.sin_io[pos_io])
        if self.d_x > 0:
            parts_cos.append(self.cos_x[pos_x])
            parts_sin.append(self.sin_x[pos_x])
        if self.d_y > 0:
            parts_cos.append(self.cos_y[pos_y])
            parts_sin.append(self.sin_y[pos_y])
        if self.d_example > 0:
            parts_cos.append(self.cos_example[pos_example])
            parts_sin.append(self.sin_example[pos_example])

        cos = (
            jnp.concatenate(parts_cos, axis=-1)
            if parts_cos
            else jnp.zeros(pos_io.shape + (0,), dtype=jnp.float32)
        )
        sin = (
            jnp.concatenate(parts_sin, axis=-1)
            if parts_sin
            else jnp.zeros(pos_io.shape + (0,), dtype=jnp.float32)
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
    attn_dropout: eqx.nn.Dropout
    proj_dropout: eqx.nn.Dropout
    rope: object
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    is_causal: bool = eqx.field(static=True)
    rope_skip: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        *,
        rope,
        is_causal: bool,
        rope_skip: int,
        dtype: jnp.dtype,
        key: jax.Array,
    ):
        self.n_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dtype = dtype
        self.dropout_rate = dropout
        self.is_causal = is_causal
        self.rope_skip = rope_skip
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
        positions: Int[Array, "B T 4"],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        B, S, E = x.shape
        inference = inference or (key is None)
        k_attn, k_proj = (None, None) if key is None else jax.random.split(key, 2)

        x = x.astype(self.dtype)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        if self.rope_skip > 0:
            q_prefix, q_main = q[:, : self.rope_skip], q[:, self.rope_skip :]
            k_prefix, k_main = k[:, : self.rope_skip], k[:, self.rope_skip :]
            pos_main = positions[:, self.rope_skip :]
            q_main, k_main = self.rope(q_main, k_main, pos_main)
            q = jnp.concatenate([q_prefix, q_main], axis=1)
            k = jnp.concatenate([k_prefix, k_main], axis=1)
        else:
            q, k = self.rope(q, k, positions)

        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        attn_out = _flash_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            is_causal=self.is_causal,
        )
        attn_out = self.attn_dropout(attn_out, key=k_attn, inference=inference)
        attn_out = attn_out.reshape(B, S, E)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out, key=k_proj, inference=inference)
        return attn_out.astype(self.dtype)

    def incremental(
        self,
        x: Float[Array, "B 1 D"],
        *,
        positions: Int[Array, "B 1 4"],
        cache: Tuple[Float[Array, "B T H D"], Float[Array, "B T H D"]],
        cache_index: Int[Array, ""],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Tuple[Float[Array, "B 1 D"], Tuple[jax.Array, jax.Array]]:
        if self.rope_skip != 0:
            raise ValueError("incremental attention does not support rope_skip")

        inference = inference or (key is None)
        k_attn, k_proj = (None, None) if key is None else jax.random.split(key, 2)

        x = x.astype(self.dtype)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(x.shape[0], 1, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q, k = self.rope(q, k, positions)
        q = q.astype(self.dtype)
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        k_cache, v_cache = cache
        cache_index = jnp.asarray(cache_index, dtype=jnp.int32)
        k_cache = lax.dynamic_update_slice(k_cache, k, (0, cache_index, 0, 0))
        v_cache = lax.dynamic_update_slice(v_cache, v, (0, cache_index, 0, 0))

        attn_out = _flash_attention(
            q,
            k_cache,
            v_cache,
            attention_mask=None,
            is_causal=self.is_causal,
            key_len=cache_index + 1,
            q_offset=cache_index,
        )
        attn_out = self.attn_dropout(attn_out, key=k_attn, inference=inference)
        attn_out = attn_out.reshape(x.shape[0], 1, -1)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out, key=k_proj, inference=inference)
        return attn_out.astype(self.dtype), (k_cache, v_cache)


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
        rope,
        is_causal: bool,
        rope_skip: int,
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
            is_causal=is_causal,
            rope_skip=rope_skip,
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
        positions: Int[Array, "B T 4"],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B T D"]:
        k_attn, k_ff = (None, None) if key is None else jax.random.split(key, 2)
        inference = inference or (key is None)

        h = self.ln1(x)
        a = self.attn(
            h,
            attention_mask=attention_mask,
            positions=positions,
            key=k_attn,
            inference=inference,
        )
        x = x + a

        h2 = self.ln2(x)
        f = self.ff(h2, key=k_ff, inference=inference)
        x = x + f
        return x.astype(self.dtype)

    def incremental(
        self,
        x: Float[Array, "B 1 D"],
        *,
        positions: Int[Array, "B 1 4"],
        cache: Tuple[Float[Array, "B T H D"], Float[Array, "B T H D"]],
        cache_index: Int[Array, ""],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Tuple[Float[Array, "B 1 D"], Tuple[jax.Array, jax.Array]]:
        k_attn, k_ff = (None, None) if key is None else jax.random.split(key, 2)
        inference = inference or (key is None)

        h = self.ln1(x)
        a, cache = self.attn.incremental(
            h,
            positions=positions,
            cache=cache,
            cache_index=cache_index,
            key=k_attn,
            inference=inference,
        )
        x = x + a

        h2 = self.ln2(x)
        f = self.ff(h2, key=k_ff, inference=inference)
        x = x + f
        return x.astype(self.dtype), cache


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
        rope_mode: str,
        rope_max_io: int,
        rope_max_x: int,
        rope_max_y: int,
        rope_max_example: int,
        is_causal: bool,
        rope_skip: int,
        dtype: jnp.dtype,
        key: jax.Array,
    ):
        if rope_mode != "4d":
            raise ValueError(f"Unsupported rope_mode: {rope_mode}")

        rope_factory = lambda: RotaryEmbedding4D(
            head_dim=embed_dim // num_heads,
            max_io=rope_max_io,
            max_x=rope_max_x,
            max_y=rope_max_y,
            max_example=rope_max_example,
        )

        keys = jax.random.split(key, depth)
        self.layers = tuple(
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_dim,
                dropout,
                rope=rope_factory(),
                is_causal=is_causal,
                rope_skip=rope_skip,
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
        positions: Int[Array, "B T 4"],
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
                positions=positions,
                key=layer_key,
                inference=inference,
            )
        return x

    def incremental(
        self,
        x: Float[Array, "B 1 D"],
        *,
        positions: Int[Array, "B 1 4"],
        caches: Tuple[Tuple[jax.Array, jax.Array], ...],
        cache_index: Int[Array, ""],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Tuple[Float[Array, "B 1 D"], Tuple[Tuple[jax.Array, jax.Array], ...]]:
        layer_keys = (
            [None] * len(self.layers)
            if key is None
            else list(jax.random.split(key, len(self.layers)))
        )
        new_caches = []
        for layer, layer_key, cache in zip(self.layers, layer_keys, caches):
            x, cache = layer.incremental(
                x,
                positions=positions,
                cache=cache,
                cache_index=cache_index,
                key=layer_key,
                inference=inference,
            )
            new_caches.append(cache)
        return x, tuple(new_caches)


def _flash_attention(
    q: Float[Array, "B T H D"],
    k: Float[Array, "B T H D"],
    v: Float[Array, "B T H D"],
    *,
    attention_mask: Optional[Bool[Array, "B T"]],
    is_causal: bool,
    key_len: Optional[Int[Array, ""]] = None,
    q_offset: Int[Array, ""] = 0,
) -> Float[Array, "B T H D"]:
    _, q_len, _, _ = q.shape
    s_len = k.shape[1]

    def _pad(x: jax.Array, pad_len: int) -> jax.Array:
        return jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    padded_q_len = ((q_len + 3) // 4) * 4
    padded_s_len = ((s_len + 3) // 4) * 4
    pad_q = padded_q_len - q_len
    pad_s = padded_s_len - s_len

    q_p = _pad(q, pad_q)
    k_p = _pad(k, pad_s)
    v_p = _pad(v, pad_s)

    base_mask = jnp.ones((1, 1, padded_q_len, padded_s_len), dtype=jnp.bool_)
    base_mask = base_mask.at[:, :, q_len:, :].set(False)
    base_mask = base_mask.at[:, :, :, s_len:].set(False)

    if key_len is not None:
        key_len = jnp.asarray(key_len, dtype=jnp.int32)
        key_keep = jnp.arange(padded_s_len) < key_len
        base_mask = base_mask & key_keep[None, None, None, :]

    if attention_mask is not None:
        key_keep = attention_mask.astype(jnp.bool_)
        if pad_s:
            key_keep = jnp.pad(key_keep, ((0, 0), (0, pad_s)), constant_values=False)
        base_mask = base_mask & key_keep[:, None, None, :]

    if is_causal:
        q_offset = jnp.asarray(q_offset, dtype=jnp.int32)
        q_positions = q_offset + jnp.arange(padded_q_len, dtype=jnp.int32)
        k_positions = jnp.arange(padded_s_len, dtype=jnp.int32)
        causal = k_positions[None, :] <= q_positions[:, None]
        base_mask = base_mask & causal[None, None, :, :]

    attn_out = jax.nn.dot_product_attention(
        query=q_p,
        key=k_p,
        value=v_p,
        mask=base_mask,
        bias=None,
        implementation="cudnn",
        is_causal=False,
    )
    return attn_out[:, :q_len, :, :]


def _rotate_half(x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    orig = x.shape
    x = x.reshape(orig[:-1] + (-1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    y = jnp.stack([-x2, x1], axis=-1)
    return y.reshape(orig)
