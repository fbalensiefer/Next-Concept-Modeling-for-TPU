"""Shared transformer scanned coarse to fine; no token autoregressive loop."""
import math

import jax
import jax.numpy as jnp
from flax import linen as nn, struct

from .config import RVQConfig, TransformerConfig
from .grids import align, grid_matrices
from .sparse_grids import sparse_align


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool

    @nn.compact
    def __call__(self, x, valid):
        attention_mask = nn.make_attention_mask(valid, valid, dtype=bool)
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.config.heads, dropout_rate=self.config.dropout,
            deterministic=self.deterministic)(h, mask=attention_mask)
        x = x + nn.Dropout(self.config.dropout)(h, deterministic=self.deterministic)
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.config.width * self.config.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.config.width)(h)
        x = x + nn.Dropout(self.config.dropout)(h, deterministic=self.deterministic)
        return jnp.where(valid[..., None], x, 0)


class LevelCell(nn.Module):
    rvq: RVQConfig
    transformer: TransformerConfig
    deterministic: bool
    temperature: float = 0.0
    top_k: int = 0
    sparse: bool = False

    @nn.compact
    def __call__(self, cumulative, xs):
        level, embeddings, down, up, valid, target, teacher_forcing, source_mask = xs
        # The current level target is read only AFTER producing logits.
        resample = sparse_align if self.sparse else align
        context, _ = resample(cumulative, source_mask, down)
        h = nn.Dense(self.transformer.width, name="context_projection")(context)
        h += nn.Embed(self.rvq.levels, self.transformer.width, name="level_embedding")(level)
        position = self.param("position", nn.initializers.normal(.02),
                              (self.rvq.max_length, self.transformer.width))
        h = jnp.where(valid[..., None], h + position[None, :context.shape[1]], 0)
        for _ in range(self.transformer.layers):
            h = TransformerBlock(self.transformer, self.deterministic)(h, valid)
        logits = nn.Dense(self.rvq.codebook_size, name="code_head")(nn.LayerNorm()(h))
        if self.temperature > 0:
            sampling_logits = logits / self.temperature
            if self.top_k:
                threshold = jax.lax.top_k(sampling_logits, self.top_k)[0][..., -1:]
                sampling_logits = jnp.where(sampling_logits >= threshold, sampling_logits, -jnp.inf)
            predicted = jax.random.categorical(self.make_rng("sample"), sampling_logits).astype(jnp.int32)
        else:
            predicted = jnp.argmax(logits, -1).astype(jnp.int32)
        selected = jnp.where(teacher_forcing, target, predicted)
        vectors = jnp.where(valid[..., None], embeddings[jnp.maximum(selected, 0)], 0)
        interpolated, _ = resample(vectors, valid, up)
        cumulative += jnp.where(source_mask[..., None], interpolated, 0)
        return cumulative, (logits, jnp.where(valid, selected, -1))


@struct.dataclass
class GeneratorOutput:
    logits: jax.Array  # [K,B,N,V]
    indices: jax.Array  # [K,B,N]
    representation: jax.Array  # [B,L,D]


class SharedLevelGenerator(nn.Module):
    rvq: RVQConfig
    transformer: TransformerConfig

    @nn.compact
    def __call__(self, codebooks, level_mask, target_codes, *,
                 teacher_forcing=True, deterministic=True, source_mask=None,
                 temperature=0.0, top_k=0):
        """Predict all positions at a level together, using earlier levels only.

        target_codes are required as a shape-stable dummy even at inference.
        level_mask [K,B,N] encodes the known bucket/padding/fallback geometry.
        teacher_forcing may be a scalar traced bool; no target gradients flow.
        """
        if level_mask.shape != target_codes.shape or level_mask.shape[0] != self.rvq.levels:
            raise ValueError("codes and mask must have matching [K,B,N] shapes")
        if level_mask.shape[2] != self.rvq.max_length:
            raise ValueError("level storage must be padded to max(lengths)")
        if not math.isfinite(temperature) or temperature < 0 or not isinstance(top_k, int) or not 0 <= top_k <= self.rvq.codebook_size:
            raise ValueError("temperature must be nonnegative and top_k within codebook size")
        down, up = grid_matrices(self.rvq)
        # Enforce structural padding even for caller-provided inference masks.
        structural = jnp.arange(self.rvq.max_length)[None, :] < jnp.asarray(self.rvq.lengths)[:, None]
        valid = level_mask & structural[:, None, :]
        # Supply the original learned-path mask during nested dropout: the
        # finest level may be dropped even when source positions are valid.
        source_mask = level_mask[-1] if source_mask is None else source_mask
        if source_mask.shape != level_mask.shape[1:]:
            raise ValueError("source_mask must have shape [B,L]")
        cell = nn.scan(LevelCell, variable_broadcast="params",
                       split_rngs={"params": False, "dropout": True, "sample": True},
                       in_axes=0, out_axes=0)(self.rvq, self.transformer, deterministic, temperature, top_k)
        cumulative = jnp.zeros((valid.shape[1], self.rvq.max_length, self.rvq.dim), jnp.float32)
        cumulative, (logits, codes) = cell(cumulative, (
            jnp.arange(self.rvq.levels), jax.lax.stop_gradient(codebooks.embeddings),
            down, up, valid, jax.lax.stop_gradient(target_codes),
            jnp.broadcast_to(jnp.asarray(teacher_forcing), (self.rvq.levels,)),
            jnp.broadcast_to(source_mask, (self.rvq.levels, *source_mask.shape))))
        return GeneratorOutput(logits, codes, cumulative)
