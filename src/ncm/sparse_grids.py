"""Linear-storage area pooling and two-point interpolation plans."""
from functools import lru_cache
import math

import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=128)
def grid_plan(size, length):
    """Actual-length plans matching grids.grid_matrices without dense matrices."""
    support = math.ceil(size / length) + 1
    starts = np.floor(np.arange(length) * size / length).astype(np.int32)
    indices = starts[:, None] + np.arange(support, dtype=np.int32)[None, :]
    left = np.arange(length)[:, None] * size / length
    right = (np.arange(length)[:, None] + 1) * size / length
    weights = np.maximum(0, np.minimum(right, indices + 1) - np.maximum(left, indices))
    weights = (weights / (size / length)).astype(np.float32)
    down = (np.clip(indices, 0, size - 1), weights)
    coordinate = np.clip((np.arange(size) + .5) * length / size - .5, 0, length - 1)
    lo = np.floor(coordinate).astype(np.int32)
    hi = np.minimum(lo + 1, length - 1)
    up = (np.stack([lo, hi], -1),
          np.stack([1 - (coordinate - lo), coordinate - lo], -1).astype(np.float32))
    return down, up


def sparse_align(values, mask, plan):
    indices, weights = map(jnp.asarray, plan)
    selected = values[:, indices, :]
    effective = weights[None] * mask[:, indices]
    mass = effective.sum(-1)
    selected = jnp.where(effective[..., None] > 0, selected, 0)
    output = (effective[..., None] * selected).sum(-2)
    return output / jnp.maximum(mass[..., None], 1e-8), mass > 0
