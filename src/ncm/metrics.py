"""Mask-aware diagnostics; no target quality or performance is assumed."""
import jax.numpy as jnp

from .rvq import reduce_stats


def code_usage(counts):
    """Global sufficient statistics -> level-wise usage and perplexity."""
    total = counts.sum(-1, keepdims=True)
    p = counts / jnp.maximum(total, 1)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-8)), 0), -1)
    return {"usage": jnp.mean(counts > 0, -1),
            "perplexity": jnp.where(total[:, 0] > 0, jnp.exp(entropy), 0)}


def exact_match(predictions, targets, mask, axis_name=None):
    """Token/sequence EM; empty examples do not count as successful sequences."""
    matches = predictions == targets
    token_hits = reduce_stats(jnp.sum(matches & mask), axis_name)
    tokens = reduce_stats(jnp.sum(mask), axis_name)
    nonempty = jnp.any(mask, -1)
    sequences = reduce_stats(jnp.sum(nonempty), axis_name)
    sequence_hits = reduce_stats(jnp.sum(jnp.all(matches | ~mask, -1) & nonempty), axis_name)
    return {"token_exact_match": token_hits / jnp.maximum(tokens, 1),
            "sequence_exact_match": sequence_hits / jnp.maximum(sequences, 1)}
