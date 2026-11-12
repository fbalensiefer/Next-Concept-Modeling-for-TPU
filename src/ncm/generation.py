"""Frozen-codec prior training and fixed-length byte generation."""
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .generator import SharedLevelGenerator
from .grids import align, grid_matrices
from .losses import cross_entropy
from .tokenizer import MultiscaleTokenizer


@struct.dataclass
class PriorState:
    params: object
    opt_state: object
    step: jax.Array


def encode(model, codec_params, codebooks, batch):
    return MultiscaleTokenizer(model.rvq, model.token_vocab_size).apply(
        {'params': codec_params}, batch.tokens, batch.mask, batch.fallback_bytes,
        batch.fallback_mask, codebooks, training=False).rvq


def make_prior_step(model, optimizer, codec_params, codebooks):
    """Only prior parameters enter autodiff/optimizer; codec and EMA are immutable."""
    generator = SharedLevelGenerator(model.rvq, model.transformer)

    def step(state, batch, key):
        target = encode(model, codec_params, codebooks, batch)

        def objective(params):
            predicted = generator.apply({'params': params}, codebooks, target.mask,
                                        target.indices, source_mask=batch.mask,
                                        deterministic=False, rngs={'dropout': key})
            return cross_entropy(predicted.logits, target.indices, target.mask)

        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        candidate = PriorState(optax.apply_updates(state.params, updates), opt_state, state.step + 1)
        return jax.lax.cond(jnp.any(batch.mask), lambda: candidate, lambda: state), loss
    return step


def level_geometry(config, mask):
    down, _ = grid_matrices(config)
    return jnp.stack([align(jnp.zeros((*mask.shape, 1)), mask, matrix)[1] for matrix in down])


def generate(model, codec_params, prior_params, codebooks, key, *, batch_size=1,
             length=None, temperature=1.0, top_k=0):
    """Generate unconditioned fixed-length bytes; temperature=0 selects greedy codes.

    Length is caller-supplied, not inferred from an evaluation target. Decode uses
    byte argmax; stochasticity is in the hierarchical codes. No EOS is modeled.
    """
    length = model.rvq.max_length if length is None else length
    if batch_size < 1 or not 1 <= length <= model.rvq.max_length:
        raise ValueError('positive batch_size and length within configured bucket required')
    mask = jnp.broadcast_to(jnp.arange(model.rvq.max_length) < length,
                            (batch_size, model.rvq.max_length))
    levels = level_geometry(model.rvq, mask)
    result = SharedLevelGenerator(model.rvq, model.transformer).apply(
        {'params': prior_params}, codebooks, levels, jnp.zeros(levels.shape, jnp.int32),
        teacher_forcing=False, source_mask=mask, temperature=temperature, top_k=top_k,
        rngs={'sample': key})
    logits = MultiscaleTokenizer(model.rvq, model.token_vocab_size).apply(
        {'params': codec_params}, result.representation, mask, method=MultiscaleTokenizer.decode)
    return jnp.where(mask, logits.argmax(-1), -1), result.indices


def bytes_to_text(tokens):
    """Display with replacement for invalid UTF-8; raw bytes remain authoritative."""
    return [bytes(int(x) for x in row if x >= 0).decode('utf-8', errors='replace')
            for row in np.asarray(tokens)]


def evaluate_prior(model, codec_params, prior_params, codebooks, batch):
    """Per-level sufficient statistics under true versus greedy generated prefixes.

    Free-prefix CE is an exposure diagnostic, not a marginal likelihood. Both
    paths use target geometry and evaluate prediction of the same target codes.
    """
    target = encode(model, codec_params, codebooks, batch)
    generator = SharedLevelGenerator(model.rvq, model.transformer)
    reports = {}
    for name, forced in [('teacher', True), ('free', False)]:
        result = generator.apply({'params': prior_params}, codebooks, target.mask,
                                 target.indices, teacher_forcing=forced, source_mask=batch.mask)
        ce = optax.softmax_cross_entropy_with_integer_labels(result.logits, jnp.maximum(target.indices, 0))
        count = target.mask.sum(axis=(1, 2))
        reports[name + '_nll_sum'] = jnp.where(target.mask, ce, 0).sum(axis=(1, 2))
        reports[name + '_hits'] = ((result.logits.argmax(-1) == target.indices) & target.mask).sum(axis=(1, 2))
        reports['count'] = count
        # Under teacher forcing the accumulated representation uses target codes:
        # its decoded accuracy is an oracle codec reconstruction ceiling.
        logits = MultiscaleTokenizer(model.rvq, model.token_vocab_size).apply(
            {'params': codec_params}, result.representation, batch.mask,
            method=MultiscaleTokenizer.decode)
        reports[name + '_byte_hits'] = ((logits.argmax(-1) == batch.tokens) & batch.mask).sum()
    reports['byte_count'] = batch.mask.sum()
    return reports
