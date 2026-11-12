"""Flax token encoder/decoder around the functional multiscale quantizer."""
import jax
import jax.numpy as jnp
from flax import linen as nn, struct

from .config import RVQConfig
from .grids import align, grid_matrices
from .rvq import RVQOutput, quantize


@struct.dataclass
class TokenizerOutput:
    logits: jax.Array
    rvq: RVQOutput
    representation: jax.Array
    coarse: jax.Array
    projected_fine: jax.Array
    concept_mask: jax.Array
    latent: jax.Array


class ResidualConv(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm()(x)
        h = jnp.where(mask[..., None], h, 0)
        h = nn.Conv(self.dim * 2, kernel_size=(3,), padding="SAME")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return jnp.where(mask[..., None], x + h, 0)


class TokenDecoder(nn.Module):
    dim: int
    vocab_size: int
    layers: int

    @nn.compact
    def __call__(self, representation, mask):
        h = jnp.where(mask[..., None], representation, 0)
        for _ in range(self.layers):
            h = ResidualConv(self.dim)(h, mask)
        return nn.Dense(self.vocab_size, name="reconstruction_head")(h)


class MultiscaleTokenizer(nn.Module):
    config: RVQConfig
    token_vocab_size: int
    encoder_layers: int = 2

    def setup(self):
        self.token_embedding = nn.Embed(self.token_vocab_size, self.config.dim)
        self.encoder = [ResidualConv(self.config.dim) for _ in range(self.encoder_layers)]
        self.byte_embedding = nn.Embed(256, self.config.dim)
        self.decoder = TokenDecoder(self.config.dim, self.token_vocab_size, self.encoder_layers)
        self.concept_projection = nn.Dense(self.config.dim)

    def encode_latent(self, tokens, padding_mask, fallback_mask):
        """Encoder-only warmup path; no quantizer or decoder work is required."""
        learned_mask = padding_mask.astype(bool) & ~fallback_mask.astype(bool)
        safe_tokens = jnp.where(learned_mask, tokens, 0)
        latent = self.token_embedding(safe_tokens)
        latent = jnp.where(learned_mask[..., None], latent, 0)
        for layer in self.encoder:
            latent = layer(latent, learned_mask)
        return latent

    def decode(self, representation, padding_mask, fallback_bytes=None, fallback_mask=None):
        """Decode a generated cumulative representation using trained weights.

        Byte metadata can route literal byte features into generated slots.
        To preserve exact bytes, retain IDs and mask in the output protocol.
        """
        if fallback_bytes is not None and fallback_mask is not None:
            selected = fallback_mask & padding_mask
            features = self.byte_embedding(jnp.where(selected, fallback_bytes, 0))
            representation = jnp.where(selected[..., None], features, representation)
        return self.decoder(representation, padding_mask)

    def __call__(self, tokens, padding_mask, fallback_bytes, fallback_mask,
                 codebooks, *, phase=0, training=False, axis_name=None, coarse_temperature=None):
        """All token inputs are [B,L]; True padding_mask means a valid token.

        fallback_bytes supplies one pre-aligned byte ID (0..255) per slot. The
        host data pipeline must expand multi-byte regions before bucketing.
        The byte IDs remain available alongside the returned representation
        for lossless literal passthrough; learned embeddings alone are not a
        lossless byte codec.
        """
        shape = tokens.shape
        if len(shape) != 2 or shape[1] != self.config.max_length:
            raise ValueError("tokens must have shape [B,max(lengths)]")
        if any(x.shape != shape for x in (padding_mask, fallback_bytes, fallback_mask)):
            raise ValueError("all tokenizer inputs must have matching [B,L] shapes")
        valid = padding_mask.astype(bool)
        fallback = fallback_mask.astype(bool) & valid
        learned_mask = valid & ~fallback
        # Fallback tokens cannot leak through encoder convolutions into codes.
        latent = self.encode_latent(tokens, valid, fallback)
        output = quantize(latent, learned_mask, codebooks, self.config,
                          phase=phase, training=training, axis_name=axis_name, coarse_temperature=coarse_temperature,
                          key=self.make_rng("dropout") if training else None)
        safe_bytes = jnp.where(fallback, fallback_bytes, 0)
        byte_features = self.byte_embedding(safe_bytes)
        representation = jnp.where(fallback[..., None], byte_features, output.quantized)
        representation = jnp.where(valid[..., None], representation, 0)
        logits = self.decode(representation, valid)
        _, up = grid_matrices(self.config)
        coarse, coarse_mask = align(output.level_vectors[0], output.mask[0], up[0])
        fine = self.concept_projection(output.quantized)
        return TokenizerOutput(logits, output, representation, coarse, fine,
                               coarse_mask & learned_mask, latent)
