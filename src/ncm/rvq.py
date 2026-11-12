"""Pure RVQ and EMA state. No codebook is hidden in a Flax variable collection."""
import jax
import jax.numpy as jnp
from flax import struct

from .config import RVQConfig
from .grids import align, grid_matrices


@struct.dataclass
class CodebookState:
    embeddings: jax.Array  # [K,V,D], float32, replicated across dp/fsdp
    counts: jax.Array  # [K,V]
    sums: jax.Array  # [K,V,D]


@struct.dataclass
class RVQOutput:
    quantized: jax.Array  # [B,L,D], straight-through estimator
    indices: jax.Array  # [K,B,N], -1 where invalid
    mask: jax.Array  # [K,B,N]
    level_vectors: jax.Array  # [K,B,N,D], hard quantized vectors
    commitment_numerator: jax.Array  # [K]
    commitment_denominator: jax.Array  # [K], number of scalar components
    counts: jax.Array  # [K,V], hard assignment statistics
    sums: jax.Array  # [K,V,D]
    soft_counts: jax.Array  # [K,V], differentiable entropy estimator


def init_codebooks(key, config: RVQConfig) -> CodebookState:
    shape = (config.levels, config.codebook_size, config.dim)
    embeddings = jax.random.normal(key, shape, jnp.float32) / config.dim**0.5
    # A unit pseudo-count keeps unused entries stable during the first updates.
    counts = jnp.ones(shape[:2], jnp.float32)
    sums = embeddings * (counts[..., None] + config.epsilon)
    return CodebookState(embeddings, counts, sums)


def reduce_stats(value, axis_name=None):
    """Use axis_name only under a bound pmap/shard_map data axis.

    With global NamedSharding arrays under jit, reductions already span global
    data and XLA inserts collectives; an additional psum would double-count.
    """
    return value if axis_name is None else jax.lax.psum(value, axis_name)


def quantize(latent, mask, state: CodebookState, config: RVQConfig, *,
             phase=0, key=None, training=False, axis_name=None, coarse_temperature=None) -> RVQOutput:
    """Scan K levels, retaining a residual on the common finest grid.

    phase=0: joint training; phase!=0: final-two-level tuning. Nested dropout
    selects a per-example prefix and is disabled dynamically in the latter.
    Inputs must have fixed shape [B,max(lengths),D] for each compiled bucket.
    """
    if latent.shape[1:] != (config.max_length, config.dim):
        raise ValueError("latent does not match the configured static grid")
    if mask.shape != latent.shape[:2]:
        raise ValueError("mask must have shape [B,L]")
    down, up = grid_matrices(config)
    batch = latent.shape[0]
    depth = jnp.full((batch,), config.levels)
    if training and config.nested_dropout:
        if key is None:
            raise ValueError("nested dropout requires a PRNG key")
        key_drop, key_depth = jax.random.split(key)
        drop = jax.random.bernoulli(key_drop, config.nested_dropout, (batch,))
        sampled = jax.random.randint(key_depth, (batch,), 1, config.levels + 1)
        depth = jnp.where((jnp.asarray(phase) == 0) & drop, sampled, depth)
    latent = jnp.where(mask[..., None], latent.astype(jnp.float32), 0)

    def level_step(cumulative, xs):
        level, embeddings, down_k, up_k = xs
        residual, valid = align(latent - jax.lax.stop_gradient(cumulative), mask, down_k)
        valid = valid & (level < depth[:, None])
        residual = jnp.where(valid[..., None], residual, 0)
        embeddings = jax.lax.stop_gradient(embeddings)
        distances = (jnp.sum(residual**2, -1, keepdims=True)
                     + jnp.sum(embeddings**2, -1)[None, None, :]
                     - 2 * jnp.einsum("bnd,vd->bnv", residual, embeddings))
        codes = jnp.argmin(distances, -1)
        vectors = jnp.where(valid[..., None], embeddings[codes], 0)
        # Scatter sufficient statistics without a [B,N,V] hard one-hot tensor.
        counts = jnp.zeros((config.codebook_size,), jnp.float32).at[codes.reshape(-1)].add(
            valid.astype(jnp.float32).reshape(-1))
        sums = jnp.zeros_like(embeddings).at[codes.reshape(-1)].add(
            residual.reshape(-1, config.dim))
        temperature = (config.entropy_temperature if coarse_temperature is None else
                       jnp.where(level == 0, coarse_temperature, config.entropy_temperature))
        soft = jax.nn.softmax(-distances / temperature, -1)
        soft_counts = jnp.sum(jnp.where(valid[..., None], soft, 0), axis=(0, 1))
        error = jnp.where(valid[..., None], (residual - vectors)**2, 0).sum()
        denom = valid.sum().astype(jnp.float32) * config.dim
        interpolated, _ = align(vectors, valid, up_k)
        cumulative = cumulative + jnp.where(mask[..., None], interpolated, 0)
        return cumulative, (jnp.where(valid, codes, -1), valid, vectors, error,
                            denom, counts, sums, soft_counts)

    cumulative, outputs = jax.lax.scan(
        level_step, jnp.zeros_like(latent),
        (jnp.arange(config.levels), state.embeddings, down, up))
    indices, valid, vectors, errors, denoms, counts, sums, soft_counts = outputs
    # Identity gradient to the encoder; no gradient to the EMA embeddings.
    straight_through = latent + jax.lax.stop_gradient(cumulative - latent)
    return RVQOutput(straight_through, indices, valid, vectors,
                     reduce_stats(errors, axis_name), reduce_stats(denoms, axis_name),
                     reduce_stats(counts, axis_name), reduce_stats(sums, axis_name),
                     reduce_stats(soft_counts, axis_name))


def ema_update(state: CodebookState, counts, sums, config: RVQConfig, *,
               phase=0, axis_name=None) -> CodebookState:
    """Reduce local stats *once*, then return a new replicated codebook state.

    Pass axis_name=None for already-reduced RVQOutput statistics. Completely
    empty levels are left unchanged (including C/S), avoiding padding drift.
    """
    counts = jax.lax.stop_gradient(reduce_stats(counts, axis_name))
    sums = jax.lax.stop_gradient(reduce_stats(sums, axis_name))
    eligible = ((jnp.asarray(phase) == 0)
                | (jnp.arange(config.levels) >= config.levels - 2))
    eligible &= counts.sum(-1) > 0
    new_counts = config.decay * state.counts + (1 - config.decay) * counts
    new_sums = config.decay * state.sums + (1 - config.decay) * sums
    embeddings = new_sums / (new_counts[..., None] + config.epsilon)
    return CodebookState(
        jnp.where(eligible[:, None, None], embeddings, state.embeddings),
        jnp.where(eligible[:, None], new_counts, state.counts),
        jnp.where(eligible[:, None, None], new_sums, state.sums))
