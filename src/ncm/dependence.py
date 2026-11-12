"""Exact finite-domain dependence diagnostics and a differentiable soft surrogate.

Enumeration is deliberately bounded; this is not a scalable TC estimator.
"""
from functools import lru_cache
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from .grids import align, grid_matrices


@lru_cache(maxsize=64)
def codewords(lengths, vocabulary, max_words=4096):
    slots = sum(lengths)
    if vocabulary < 2 or slots < 1 or vocabulary**slots > max_words:
        raise ValueError('Code domain exceeds exact enumeration budget')
    return np.asarray(list(itertools.product(range(vocabulary), repeat=slots)), np.int32)


def entropy(probabilities):
    return -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-30)))


def conditional_tc(joint, lengths, vocabulary):
    """TC(C_k | C_<k), in nats, computed by entropy identities."""
    table = joint.reshape((vocabulary,) * sum(lengths))
    def marginal_entropy(kept):
        axes = tuple(i for i in range(sum(lengths)) if i not in kept)
        return entropy(table.sum(axis=axes))
    start = 0
    result = []
    for count in lengths:
        prefix = tuple(range(start))
        value = sum(marginal_entropy(prefix + (start + j,)) for j in range(count))
        value -= marginal_entropy(tuple(range(start + count)))
        value -= (count - 1) * marginal_entropy(prefix)
        result.append(value)
        start += count
    return jnp.stack(result)


def soft_joint(probabilities, weights, lengths, vocabulary):
    """Mixture of per-input independent soft assignments, not hard-code TC."""
    words = codewords(tuple(lengths), vocabulary)
    factors = probabilities[:, jnp.arange(sum(lengths))[None, :], jnp.asarray(words)]
    weights = weights / jnp.maximum(weights.sum(), 1e-30)
    return jnp.sum(weights[:, None] * jnp.prod(factors, axis=-1), axis=0)


def hard_joint(codes, weights, lengths, vocabulary):
    words = codewords(tuple(lengths), vocabulary)
    del words  # enforce the same enumeration bound
    powers = jnp.asarray(vocabulary**np.arange(sum(lengths) - 1, -1, -1), jnp.int32)
    indices = (codes * powers).sum(-1)
    return jnp.zeros((vocabulary**sum(lengths),), jnp.float32).at[indices].add(weights / weights.sum())


def soft_assignments(latent, books, config, temperature, coarse_temperature=None):
    """Soft posteriors around the same hard residual trajectory as quantize()."""
    down, up = grid_matrices(config)
    mask = jnp.ones(latent.shape[:2], bool)
    cumulative = jnp.zeros_like(latent)
    probabilities = []
    for level, count in enumerate(config.lengths):
        residual, valid = align(latent - jax.lax.stop_gradient(cumulative), mask, down[level, :count])
        embeddings = jax.lax.stop_gradient(jnp.asarray(books.embeddings)[level])
        distances = ((residual**2).sum(-1, keepdims=True) + (embeddings**2).sum(-1)[None, None]
                     - 2 * jnp.einsum('bnd,vd->bnv', residual, embeddings))
        level_temperature = coarse_temperature if level == 0 and coarse_temperature is not None else temperature
        probabilities.append(jax.nn.softmax(-distances / level_temperature, axis=-1))
        vectors = embeddings[jnp.argmin(distances, -1)]
        interpolated, _ = align(vectors, valid, up[level, :, :count])
        cumulative += interpolated
    return jnp.concatenate(probabilities, axis=1)


def direct_factorization_kl(joint, lengths, vocabulary):
    """Independent NumPy chain-rule construction to verify the TC identity."""
    words = codewords(tuple(lengths), vocabulary)
    p = np.asarray(joint, np.float64)
    q = np.ones(len(words), np.float64)
    offset = 0
    for count in lengths:
        for i, word in enumerate(words):
            context = np.all(words[:, :offset] == word[:offset], axis=1)
            denominator = p[context].sum()
            for j in range(offset, offset + count):
                numerator = p[context & (words[:, j] == word[j])].sum()
                q[i] *= numerator / denominator if denominator > 0 else 1 / vocabulary
        offset += count
    positive = p > 0
    return float(np.sum(p[positive] * np.log(p[positive] / np.maximum(q[positive], 1e-300))))
