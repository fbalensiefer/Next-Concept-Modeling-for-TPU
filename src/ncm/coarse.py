"""Deterministic coarse VQ initialization, usage recovery and optimizer group.

CodebookState remains unchanged. All additional buffers live in optimizer state,
so historical (disabled) checkpoints keep their existing schema.
"""
from flax import struct
import jax
import jax.numpy as jnp
import optax

from .grids import align, grid_matrices
from .rvq import ema_update


@struct.dataclass
class CoarseState:
    initialized: jax.Array
    step: jax.Array
    initial_samples: jax.Array
    hard_counts: jax.Array
    replacements: jax.Array
    base_key: jax.Array  # serializable legacy uint32[2], not typed key<fry>
    optimizer_state: object
    monitored_steps: jax.Array
    collapsed_steps: jax.Array
    min_hard_perplexity: jax.Array


def centroid_optimizer(config):
    # Separate from neural Adam: the pseudo-gradient is E - EMA_centroid.
    # Fine books keep the ordinary EMA update, equivalent to centroid rate 1.
    return optax.multi_transform(
        {'coarse_codebook': optax.sgd(config.coarse_codebook_lr_scale)},
        {'embedding': 'coarse_codebook'})


def new_coarse_state(books, config, key, step=0):
    optimizer = centroid_optimizer(config)
    return CoarseState(jnp.asarray(False), jnp.asarray(step, jnp.int32), jnp.asarray(0, jnp.int32),
                       jnp.zeros_like(books.counts[0]), jnp.zeros(books.counts.shape[1], jnp.int32),
                       jax.random.key_data(key), optimizer.init({'embedding': books.embeddings[0]}),
                       jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
                       jnp.asarray(books.counts.shape[1], jnp.float32))


def linear_schedule(start, end, steps, step):
    fraction = jnp.minimum(jnp.asarray(step, jnp.float32) / max(steps, 1), 1.)
    fraction = jnp.where(steps == 0, 1., fraction)
    return jnp.asarray(start, jnp.float32) + fraction * (end - start)


def schedules(config, state):
    return (linear_schedule(config.temp_start, config.temp_end, config.temp_anneal_steps, state.step),
            linear_schedule(config.beta_start, config.beta_end, config.beta_warmup_steps, state.step))


def coarse_points(latent, mask, rvq):
    down, _ = grid_matrices(rvq)
    values, valid = align(latent, mask, down[0, :rvq.lengths[0]])
    return values.reshape(-1, rvq.dim), valid.reshape(-1)


def kmeans(points, valid, size, key, *, iterations=20, noise=1e-4):
    """Masked K-means++ and fixed-count Lloyd iterations, with seeded empty centers.

    Noise prevents identical fallback centers on degenerate/undersized data;
    it cannot make an actually constant encoder use multiple hard codes.
    """
    points = jnp.where(valid[:, None], points.astype(jnp.float32), 0)
    uniform_logits = jnp.where(valid, 0., -jnp.inf)
    first = jax.random.categorical(jax.random.fold_in(key, 0), uniform_logits)
    centers = jnp.zeros((size, points.shape[-1]), jnp.float32).at[0].set(points[first])
    nearest = jnp.sum((points - centers[0])**2, -1)
    def seed(i, carry):
        centers, nearest = carry
        mass = jnp.where(valid, nearest, 0)
        logits = jnp.where(valid & (mass > 0), jnp.log(jnp.maximum(mass, 1e-30)), -jnp.inf)
        logits = jnp.where(mass.sum() > 0, logits, uniform_logits)
        chosen = jax.random.categorical(jax.random.fold_in(key, i), logits)
        jitter = noise * jax.random.normal(jax.random.fold_in(key, i + size), (points.shape[-1],))
        center = points[chosen] + jnp.where(nearest[chosen] > 0, 0., jitter)
        return centers.at[i].set(center), jnp.minimum(nearest, jnp.sum((points - center)**2, -1))
    centers, _ = jax.lax.fori_loop(1, size, seed, (centers, nearest))
    def lloyd(_, centers):
        distance = jnp.sum((points[:, None] - centers[None])**2, -1)
        code = jnp.argmin(distance, -1)
        count = jnp.zeros(size, jnp.float32).at[code].add(valid.astype(jnp.float32))
        total = jnp.zeros_like(centers).at[code].add(jnp.where(valid[:, None], points, 0))
        return jnp.where(count[:, None] > 0, total / jnp.maximum(count[:, None], 1), centers)
    centers = jax.lax.fori_loop(0, iterations, lloyd, centers)
    centers += noise * jax.random.normal(jax.random.fold_in(key, 2 * size), centers.shape)
    codes = jnp.argmin(jnp.sum((points[:, None] - centers[None])**2, -1), -1)
    counts = jnp.zeros(size, jnp.float32).at[codes].add(valid.astype(jnp.float32))
    return centers, counts


def initialize_coarse(books, state, latent, mask, rvq, *, batches=1):
    """Fit once on initial frozen-encoder outputs. Re-entry preserves all state."""
    config = rvq.coarse
    def fit(_):
        points, valid = coarse_points(latent, mask, rvq)
        centers, counts = kmeans(points, valid, rvq.codebook_size,
                                 jax.random.fold_in(state.base_key, 101),
                                 iterations=config.kmeans_iterations, noise=config.initialization_noise)
        usage = counts / batches
        effective_count = jnp.maximum(usage, config.reset_count)
        candidate = books.replace(embeddings=books.embeddings.at[0].set(centers),
                                  counts=books.counts.at[0].set(effective_count),
                                  sums=books.sums.at[0].set(centers * (effective_count[:, None] + rvq.epsilon)))
        initialized = state.replace(initialized=jnp.any(valid), initial_samples=valid.sum().astype(jnp.int32),
                                    hard_counts=usage)
        return jax.lax.cond(jnp.any(valid), lambda: (candidate, initialized), lambda: (books, state))
    return jax.lax.cond(state.initialized, lambda _: (books, state), fit, operand=None)


def maintain_coarse(books, state, counts, sums, latent, mask, rvq, *, phase=0):
    """EMA statistics + coarse centroid optimizer + branchless dead-code recovery.

    Counts are already global. Candidate sampling/noise runs even with no dead
    codes, so every arm executes the same maintenance graph and update budget.
    """
    config = rvq.coarse
    proposal = ema_update(books, counts, sums, rvq, phase=phase)
    params = {'embedding': books.embeddings[0]}
    pseudo_grads = {'embedding': books.embeddings[0] - proposal.embeddings[0]}
    optimizer = centroid_optimizer(config)
    updates, optimizer_state = optimizer.update(pseudo_grads, state.optimizer_state, params)
    embedding = optax.apply_updates(params, updates)['embedding']
    eligible = ((jnp.asarray(phase) == 0) | (rvq.levels <= 2)) & (counts[0].sum() > 0)
    usage = config.usage_decay * state.hard_counts + (1 - config.usage_decay) * counts[0]
    points, valid = coarse_points(jax.lax.stop_gradient(latent), mask, rvq)
    key = jax.random.fold_in(state.base_key, state.step)
    candidates = jax.random.categorical(jax.random.fold_in(key, 102), jnp.where(valid, 0., -jnp.inf),
                                        shape=(rvq.codebook_size,))
    replacement = points[candidates] + config.replacement_noise * jax.random.normal(
        jax.random.fold_in(key, 103), embedding.shape)
    dead = ((usage < config.dead_code_threshold) & (state.step >= config.dead_code_grace_steps)
            & (state.step % config.dead_code_check_every == 0) & jnp.any(valid) & eligible)
    reset = max(config.reset_count, 2 * config.dead_code_threshold)
    embedding = jnp.where(dead[:, None], replacement, embedding)
    new_counts = jnp.where(dead, reset, proposal.counts[0])
    new_sums = jnp.where(dead[:, None], replacement * (reset + rvq.epsilon), proposal.sums[0])
    result = proposal.replace(
        embeddings=proposal.embeddings.at[0].set(jnp.where(eligible, embedding, books.embeddings[0])),
        counts=proposal.counts.at[0].set(jnp.where(eligible, new_counts, books.counts[0])),
        sums=proposal.sums.at[0].set(jnp.where(eligible, new_sums, books.sums[0])))
    updated = state.replace(hard_counts=jnp.where(eligible, jnp.where(dead, reset, usage), state.hard_counts),
                            replacements=state.replacements + dead.astype(jnp.int32),
                            step=state.step + 1, optimizer_state=optimizer_state)
    probability = counts[0] / jnp.maximum(counts[0].sum(), 1)
    entropy = -jnp.sum(probability * jnp.log(jnp.maximum(probability, 1e-30)))
    perplexity = jnp.where(counts[0].sum() > 0, jnp.exp(entropy), 0)
    watch = eligible & (state.step >= max(config.temp_anneal_steps, config.beta_warmup_steps,
                                          config.dead_code_grace_steps))
    updated = updated.replace(monitored_steps=state.monitored_steps + watch.astype(jnp.int32),
                              collapsed_steps=state.collapsed_steps + (watch & (jnp.sum(counts[0] > 0) < rvq.codebook_size)).astype(jnp.int32),
                              min_hard_perplexity=jnp.where(watch, jnp.minimum(state.min_hard_perplexity, perplexity), state.min_hard_perplexity))
    metrics = {'coarse_active_codes': (counts[0] > 0).sum(),
               'coarse_hard_perplexity': perplexity,
               'coarse_dead_replacements': dead.sum(), 'coarse_usage_min': updated.hard_counts.min()}
    return result, updated, metrics


def optimizer_state(neural_state, books, rvq, key):
    if not rvq.coarse.enabled:
        return neural_state
    return {'neural': neural_state, 'coarse': new_coarse_state(books, rvq.coarse, key)}


def warmup_codebooks(tokenizer, params, books, state, batches):
    """Collect fixed initial batches without training the encoder, then fit once."""
    latent, masks = [], []
    for batch in batches:
        latent.append(tokenizer.apply({'params': params}, batch.tokens, batch.mask, batch.fallback_mask,
                                      method=tokenizer.encode_latent))
        masks.append(batch.mask & ~batch.fallback_mask)
    if not latent:
        raise ValueError('Coarse initialization requires at least one warmup batch')
    return initialize_coarse(books, state, jnp.concatenate(latent), jnp.concatenate(masks),
                             tokenizer.config, batches=len(batches))


def ensure_initialized(tokenizer, params, books, state, batch):
    """Recover safely if initialization previously saw only padding/fallback."""
    def fit(_):
        latent = tokenizer.apply({'params': params}, batch.tokens, batch.mask, batch.fallback_mask,
                                 method=tokenizer.encode_latent)
        return initialize_coarse(books, state, latent, batch.mask & ~batch.fallback_mask, tokenizer.config)
    return jax.lax.cond(state.initialized, lambda _: (books, state), fit, operand=None)
