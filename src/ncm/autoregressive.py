"""Small causal byte Transformer with an explicit KV cache for generation."""
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import TransformerConfig


class CausalBlock(nn.Module):
    config: TransformerConfig

    def setup(self):
        width = self.config.width
        self.attention_norm = nn.LayerNorm()
        self.qkv = nn.Dense(3 * width)
        self.attention_out = nn.Dense(width)
        self.mlp_norm = nn.LayerNorm()
        self.expand = nn.Dense(width * self.config.mlp_ratio)
        self.contract = nn.Dense(width)

    def projections(self, x):
        return self.qkv(self.attention_norm(x)).reshape(
            (*x.shape[:-1], 3, self.config.heads, self.config.width // self.config.heads))

    def finish(self, x, attended):
        x = x + self.attention_out(attended.reshape(x.shape))
        return x + self.contract(nn.gelu(self.expand(self.mlp_norm(x))))

    def __call__(self, x, valid, causal=True):
        qkv = self.projections(x)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        logits = jnp.einsum('bthd,bshd->bhts', q, k) / q.shape[-1]**.5
        causal_enabled = causal
        causal = jnp.arange(x.shape[1])[:, None] >= jnp.arange(x.shape[1])[None, :]
        mask = (causal[None, None] if causal_enabled else True) & valid[:, None, None, :]
        probabilities = jax.nn.softmax(jnp.where(mask, logits, -1e30), -1)
        attended = jnp.einsum('bhts,bshd->bthd', probabilities, v)
        return jnp.where(valid[..., None], self.finish(x, attended), 0)

    def decode(self, x, position, cache):
        qkv = self.projections(x)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        cached_k = jax.lax.dynamic_update_slice(cache[0], k[:, None], (0, position, 0, 0))
        cached_v = jax.lax.dynamic_update_slice(cache[1], v[:, None], (0, position, 0, 0))
        logits = jnp.einsum('bhd,bshd->bhs', q, cached_k) / q.shape[-1]**.5
        probabilities = jax.nn.softmax(jnp.where(
            jnp.arange(cached_k.shape[1])[None, None] <= position, logits, -1e30), -1)
        attended = jnp.einsum('bhs,bshd->bhd', probabilities, cached_v)
        return self.finish(x, attended), (cached_k, cached_v)


class AutoregressiveBytes(nn.Module):
    config: TransformerConfig
    max_length: int

    def setup(self):
        self.embedding = nn.Embed(257, self.config.width)  # 256 is BOS, never an output byte.
        self.position = self.param('position', nn.initializers.normal(.02),
                                   (self.max_length, self.config.width))
        self.blocks = [CausalBlock(self.config) for _ in range(self.config.layers)]
        self.norm = nn.LayerNorm()
        self.head = nn.Dense(256)

    def represent(self, inputs, valid):
        """Normalized causal features, before the byte output head."""
        h = self.embedding(inputs) + self.position[None, :inputs.shape[1]]
        for block in self.blocks:
            h = block(h, valid)
        return self.norm(h)

    def __call__(self, inputs, valid):
        return self.head(self.represent(inputs, valid))

    def decode_step(self, token, position, cache):
        h = self.embedding(token) + jnp.asarray(self.position)[position]
        updated = []
        for block, values in zip(self.blocks, cache):
            h, values = block.decode(h, position, values)
            updated.append(values)
        return self.head(self.norm(h)), tuple(updated)


def shifted_inputs(tokens):
    return jnp.concatenate([jnp.full((tokens.shape[0], 1), 256, jnp.int32), tokens[:, :-1]], -1)


def sample_ar(model, params, key, *, batch_size, length, temperature=1., top_k=0):
    if not 1 <= length <= model.max_length or batch_size < 1:
        raise ValueError('invalid batch_size or length')
    if (not math.isfinite(temperature) or temperature < 0 or not isinstance(top_k, int)
            or not 0 <= top_k <= 256):
        raise ValueError('invalid temperature or top_k')
    shape = (batch_size, length, model.config.heads, model.config.width // model.config.heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(model.config.layers))

    def step(carry, position):
        previous, cache = carry
        logits, cache = model.apply({'params': params}, previous, position, cache,
                                    method=model.decode_step)
        if temperature > 0:
            logits = logits / temperature
            if top_k:
                cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
                logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
            token = jax.random.categorical(jax.random.fold_in(key, position), logits).astype(jnp.int32)
        else:
            token = logits.argmax(-1).astype(jnp.int32)
        return (token, cache), token
    _, tokens = jax.lax.scan(step, (jnp.full((batch_size,), 256, jnp.int32), cache), jnp.arange(length))
    return tokens.T
