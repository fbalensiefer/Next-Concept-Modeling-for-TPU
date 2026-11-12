"""Training-only balanced decoder replay for frozen finite-domain codecs.

No teacher probabilities or evaluation examples enter the repair objective.
"""
import jax
import jax.numpy as jnp
import optax
from flax import struct

from .generation import PriorState
from .losses import cross_entropy
from .tokenizer import TokenDecoder


@struct.dataclass
class FrozenCodec:
    params: object
    codebooks: object


def make_decoder_repair_step(dim, optimizer, layers=2):
    decoder = TokenDecoder(dim, 256, layers)

    def step(state: PriorState, representations, tokens):
        mask = jnp.ones(tokens.shape, bool)
        def objective(params):
            logits = decoder.apply({'params': params}, jax.lax.stop_gradient(representations), mask)
            return cross_entropy(logits, tokens, mask)
        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        return PriorState(optax.apply_updates(state.params, updates), opt_state, state.step + 1), loss
    return step


def codec_training_distribution(train_ids, population_size, mode='empirical'):
    """Return sampler support and training-frequency weights for dependence.

    Balanced reconstruction uses uniform observed sequences. Self-normalized
    importance weights keep the soft-TC mixture aimed at empirical teacher mass.
    """
    import numpy as np
    train_ids = np.asarray(train_ids, np.int32)
    if train_ids.size == 0:
        raise ValueError('Training observations must be nonempty')
    if mode == 'empirical':
        return train_ids, np.ones(population_size, np.float32)
    if mode != 'balanced_observed':
        raise ValueError('Unsupported codec sampling mode')
    distinct, counts = np.unique(train_ids, return_counts=True)
    weights = np.zeros(population_size, np.float32)
    weights[distinct] = counts * len(distinct) / len(train_ids)
    return distinct, weights


def encoding_diagnostics(codes, probabilities):
    """Deterministic code-to-sequence fidelity ceiling on a fixed finite domain."""
    import numpy as np
    codes = np.asarray(codes)
    probabilities = np.asarray(probabilities, np.float64)
    unique, inverse = np.unique(codes, axis=0, return_inverse=True)
    best = np.zeros(len(unique), np.float64)
    np.maximum.at(best, inverse, probabilities)
    groups = [np.flatnonzero(inverse == group).tolist() for group in range(len(unique))]
    return {'population_unique_encodings': len(unique),
            'max_sequence_reconstruction_from_codes': float(best.sum()),
            'code_collision_groups': [group for group in groups if len(group) > 1]}
