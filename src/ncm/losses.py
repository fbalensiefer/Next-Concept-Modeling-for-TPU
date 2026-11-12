"""Padding-safe float32 objectives; empty batches contribute exactly zero."""
import jax
import jax.numpy as jnp
import optax

from .config import LossConfig
from .rvq import RVQOutput, reduce_stats


def masked_mean(values, mask, axis_name=None):
    numerator = reduce_stats(jnp.where(mask, values.astype(jnp.float32), 0).sum(), axis_name)
    denominator = reduce_stats(jnp.broadcast_to(mask, values.shape).sum(), axis_name)
    return numerator / jnp.maximum(denominator, 1)


def cross_entropy(logits, targets, mask, axis_name=None):
    """Aligned classification targets. Negative sentinel targets are ignored."""
    valid = mask & (targets >= 0) & (targets < logits.shape[-1])
    safe_logits = jnp.where(valid[..., None], logits.astype(jnp.float32), 0)
    safe_targets = jnp.where(valid, targets, 0)
    loss = optax.softmax_cross_entropy_with_integer_labels(safe_logits, safe_targets)
    return masked_mean(loss, valid, axis_name)


def next_token_cross_entropy(logits, tokens, mask, axis_name=None):
    """For an optional token-causal decoder, only valid adjacent pairs count.

    The NCM prior instead uses aligned next-*level* targets in cross_entropy;
    shifting those within a level would introduce the wrong training objective.
    """
    return cross_entropy(logits[:, :-1], tokens[:, 1:],
                         mask[:, :-1] & mask[:, 1:], axis_name)


def entropy_loss(soft_counts, weights, epsilon=1e-5):
    """Negative weighted entropy. Soft counts provide an encoder gradient."""
    total = soft_counts.sum(-1, keepdims=True)
    probabilities = soft_counts / jnp.maximum(total, 1e-8)
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + epsilon), -1)
    entropy = jnp.where(total[:, 0] > 0, entropy, 0)
    return -jnp.sum(weights * entropy)


def concept_consistency(coarse, projected_fine, mask, axis_name=None):
    """Features must already be aligned to the same grid."""
    coarse = jnp.where(mask[..., None], coarse.astype(jnp.float32), 0)
    fine = jnp.where(mask[..., None], projected_fine.astype(jnp.float32), 0)
    dot = jnp.sum(coarse * fine, -1)
    norm = jnp.sqrt(jnp.sum(coarse**2, -1) + 1e-8)
    norm *= jnp.sqrt(jnp.sum(fine**2, -1) + 1e-8)
    return masked_mean(1 - dot / norm, mask, axis_name)


def combined_loss(reconstruction_logits, tokens, token_mask, rvq: RVQOutput,
                  prior_logits, config: LossConfig, *, coarse=None,
                  projected_fine=None, concept_mask=None, axis_name=None,
                  coarse_beta=None, coarse_entropy_weight=0.0):
    """RVQ statistics are already global; other terms reduce their own masks."""
    levels = rvq.counts.shape[0]
    if config.entropy_weights and len(config.entropy_weights) != levels:
        raise ValueError("entropy_weights must have one value per level")
    weights = jnp.asarray(config.entropy_weights or tuple(range(levels, 0, -1)))
    recon = cross_entropy(reconstruction_logits, tokens, token_mask, axis_name)
    commit = (rvq.commitment_numerator.sum()
              / jnp.maximum(rvq.commitment_denominator.sum(), 1))
    entropy = entropy_loss(rvq.soft_counts, weights)
    prior = cross_entropy(prior_logits, rvq.indices, rvq.mask, axis_name)
    concept = jnp.asarray(0., jnp.float32)
    if config.concept:
        if coarse is None or projected_fine is None or concept_mask is None:
            raise ValueError("concept loss requires aligned features and a mask")
        concept = concept_consistency(coarse, projected_fine, concept_mask, axis_name)
    weighted_commitment = config.beta * commit
    if coarse_beta is not None:
        beta = jnp.full((levels,), config.beta).at[0].set(coarse_beta)
        weighted_commitment = (beta * rvq.commitment_numerator).sum() / jnp.maximum(rvq.commitment_denominator.sum(), 1)
    coarse_entropy = entropy_loss(rvq.soft_counts[:1], jnp.ones(1))
    total = (recon + weighted_commitment + config.entropy * entropy
             + coarse_entropy_weight * coarse_entropy
             + config.concept * concept + config.prior * prior)
    return total, dict(loss=total, reconstruction=recon, commitment=commit,
                      entropy=entropy, concept=concept, prior=prior)
