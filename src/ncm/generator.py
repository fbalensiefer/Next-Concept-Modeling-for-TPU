"""Shared coarse-to-fine Transformer with optional prompt-conditioned coarse AR."""
import math

import jax
import jax.numpy as jnp
from flax import linen as nn, struct

from .config import RVQConfig, TransformerConfig, NCMConfig
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
    config: NCMConfig = NCMConfig()

    @nn.compact
    def __call__(self, codebooks, level_mask, target_codes, *,
                 teacher_forcing=True, deterministic=True, source_mask=None,
                 temperature=0.0, top_k=0, prompt_ids=None, prompt_embeddings=None, prompt_mask=None):
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
        if self.config.ar_coarse_levels or self.config.prompt_len:
            return HybridGenerator(self.rvq, self.transformer, self.config)(
                codebooks, level_mask, target_codes, teacher_forcing=teacher_forcing,
                source_mask=source_mask, temperature=temperature, top_k=top_k,
                prompt_ids=prompt_ids, prompt_embeddings=prompt_embeddings, prompt_mask=prompt_mask)
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


class HybridGenerator(nn.Module):
    """Actual-resolution coarse AR grids with shared attention/MLP weights.

    Prompt states precede grid states. Causal masking prevents prompt states
    from reading grid states; parallel levels attend to the full prompt.
    """
    rvq: RVQConfig
    transformer: TransformerConfig
    config: NCMConfig

    @nn.compact
    def __call__(self, books, masks, targets, *, teacher_forcing=True,
                 source_mask=None, temperature=0., top_k=0,
                 prompt_ids=None, prompt_embeddings=None, prompt_mask=None):
        from .autoregressive import CausalBlock
        if self.config.ar_coarse_levels > self.rvq.levels:
            raise ValueError('ar_coarse_levels exceeds RVQ levels')
        b, storage = masks.shape[1:]
        source_mask = masks[-1] if source_mask is None else source_mask
        if prompt_embeddings is not None and prompt_ids is not None:
            raise ValueError('supply prompt IDs or embeddings, not both')
        if prompt_embeddings is None:
            ids = jnp.zeros((b, self.config.prompt_len), jnp.int32) if prompt_ids is None else prompt_ids
            if ids.shape != (b, self.config.prompt_len):
                raise ValueError('prompt IDs must match configured bucket')
            prompt = nn.Embed(257, self.transformer.width, name='prompt_embedding')(ids)
        else:
            if prompt_embeddings.shape != (b, self.config.prompt_len, self.transformer.width):
                raise ValueError('prompt embeddings must match configured bucket and width')
            prompt = prompt_embeddings
        pm = jnp.ones((b, self.config.prompt_len), bool) if prompt_mask is None else prompt_mask
        if pm.shape != prompt.shape[:2]:
            raise ValueError('prompt mask must match prompt bucket')
        prompt_position = self.param('prompt_position', nn.initializers.normal(.02),
                                     (self.config.prompt_len, self.transformer.width))
        prompt = jnp.where(pm[..., None], prompt + prompt_position[None], 0)
        projection = nn.Dense(self.transformer.width, name='context_projection')
        code_embed = nn.Embed(self.rvq.codebook_size + 1, self.transformer.width, name='code_embedding')
        level_embed = nn.Embed(self.rvq.levels, self.transformer.width, name='level_embedding')
        position = self.param('position', nn.initializers.normal(.02), (storage, self.transformer.width))
        blocks = [CausalBlock(self.transformer, name=f'block_{i}') for i in range(self.transformer.layers)]
        norm = nn.LayerNorm(name='norm')
        head = nn.Dense(self.rvq.codebook_size, name='head')
        down, up = grid_matrices(self.rvq)
        cumulative = jnp.zeros((b, storage, self.rvq.dim))
        outputs, indices = [], []
        for k, n in enumerate(self.rvq.lengths):
            valid = masks[k, :, :n]
            context, _ = align(cumulative, source_mask, down[k, :n])
            base = projection(context) + level_embed(jnp.asarray([k]))[0] + position[None, :n]
            causal = k < self.config.ar_coarse_levels
            def predict(codes):
                h = base
                if causal:
                    shifted = jnp.concatenate([jnp.full((b, 1), self.rvq.codebook_size),
                                               jnp.where(valid[:, :-1], jnp.maximum(codes[:, :-1], 0), 0)], 1)
                    h = h + code_embed(shifted)
                    h = jnp.concatenate([prompt, h], 1)
                    mask = jnp.concatenate([pm, valid], 1)
                else:
                    h = jnp.concatenate([prompt, h], 1)
                    mask = jnp.concatenate([pm, valid], 1)
                for block in blocks:
                    h = block(h, mask, causal=causal)
                return head(norm(h[:, self.config.prompt_len:]))
            def draw(logits, step):
                if temperature == 0:
                    return logits.argmax(-1).astype(jnp.int32)
                logits = logits / temperature
                if top_k:
                    cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
                    logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
                return jax.random.categorical(jax.random.fold_in(level_key, step), logits).astype(jnp.int32)
            level_key = self.make_rng('sample') if temperature else jax.random.key(0)
            # Initialize all shared Flax parameters outside a raw JAX loop.
            forced_logits = predict(targets[k, :, :n])
            if causal:
                def decode(_):
                    def step(i, codes):
                        logits = predict(codes)
                        token = draw(logits[:, i], i)
                        return codes.at[:, i].set(jnp.where(valid[:, i], token, 0))
                    codes = jax.lax.fori_loop(0, n, step, jnp.zeros((b, n), jnp.int32))
                    return predict(codes), codes
                logits, selected = jax.lax.cond(jnp.asarray(teacher_forcing),
                    lambda _: (forced_logits, targets[k, :, :n]), decode, operand=None)
            else:
                logits = forced_logits
                selected = jnp.where(teacher_forcing, targets[k, :, :n], draw(logits, 0))
            vectors = jnp.where(valid[..., None], books.embeddings[k][jnp.maximum(selected, 0)], 0)
            interpolated, _ = align(vectors, valid, up[k, :, :n])
            cumulative += jnp.where(source_mask[..., None], interpolated, 0)
            outputs.append(jnp.pad(logits, ((0, 0), (0, storage-n), (0, 0))))
            indices.append(jnp.pad(jnp.where(valid, selected, -1), ((0, 0), (0, storage-n)), constant_values=-1))
        return GeneratorOutput(jnp.stack(outputs), jnp.stack(indices), cumulative)
