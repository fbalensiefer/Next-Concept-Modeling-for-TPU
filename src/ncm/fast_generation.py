"""Inference using actual level lengths, shared weights and sparse resampling."""
import math

import jax
import jax.numpy as jnp

from .generator import LevelCell
from .sparse_grids import grid_plan, sparse_align
from .tokenizer import MultiscaleTokenizer


def generate_fast(model, codec_params, prior_params, codebooks, key, *, batch_size=1,
                  length=None, temperature=1., top_k=0):
    length = model.rvq.max_length if length is None else length
    if batch_size < 1 or not 1 <= length <= model.rvq.max_length:
        raise ValueError('positive batch_size and length within configured bucket required')
    if (not math.isfinite(temperature) or temperature < 0 or not isinstance(top_k, int)
            or not 0 <= top_k <= model.rvq.codebook_size):
        raise ValueError('invalid temperature or top_k')
    size = model.rvq.max_length
    mask = jnp.broadcast_to(jnp.arange(size) < length, (batch_size, size))
    cumulative = jnp.zeros((batch_size, size, model.rvq.dim), jnp.float32)
    cell = LevelCell(model.rvq, model.transformer, True, temperature, top_k, sparse=True)
    embeddings = jnp.asarray(codebooks.embeddings)
    codes = []
    for level, count in enumerate(model.rvq.lengths):
        down, up = grid_plan(size, count)
        _, valid = sparse_align(cumulative, mask, down)
        cumulative, (_, indices) = cell.apply(
            {'params': prior_params['ScanLevelCell_0']}, cumulative,
            (jnp.asarray(level), embeddings[level], down, up, valid,
             jnp.zeros(valid.shape, jnp.int32), jnp.asarray(False), mask),
            rngs={'sample': jax.random.fold_in(key, level)})
        codes.append(jnp.pad(indices, ((0, 0), (0, size - count)), constant_values=-1))
    logits = MultiscaleTokenizer(model.rvq, model.token_vocab_size).apply(
        {'params': codec_params}, cumulative, mask, method=MultiscaleTokenizer.decode)
    return jnp.where(mask, logits.argmax(-1), -1), jnp.stack(codes)
