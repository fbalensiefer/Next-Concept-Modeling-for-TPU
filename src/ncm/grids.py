"""Static, mask-aware 1D grid alignment for variable-length level buckets."""
import jax.numpy as jnp
import numpy as np

from .config import RVQConfig


def grid_matrices(config: RVQConfig):
    """Return padded area-pooling [K,N,L] and interpolation [K,L,N].

    L = N = max(lengths). Area pooling handles non-divisible lengths. Linear
    interpolation uses cell centers and constant boundary extension.
    """
    size = config.max_length
    downs, ups = [], []
    for length in config.lengths:
        left = np.arange(length)[:, None] * size / length
        right = (np.arange(length)[:, None] + 1) * size / length
        overlap = np.maximum(0, np.minimum(right, np.arange(size) + 1)
                             - np.maximum(left, np.arange(size)))
        down = np.zeros((size, size), np.float32)
        down[:length] = overlap / (size / length)
        coord = np.clip((np.arange(size) + .5) * length / size - .5, 0, length - 1)
        lo = np.floor(coord).astype(int)
        hi = np.minimum(lo + 1, length - 1)
        up = np.zeros((size, size), np.float32)
        up[np.arange(size), lo] += 1 - (coord - lo)
        up[np.arange(size), hi] += coord - lo
        downs.append(down)
        ups.append(up)
    return jnp.asarray(np.stack(downs)), jnp.asarray(np.stack(ups))


def align(values, mask, matrix):
    """Resample [B,S,D] with [T,S]; invalid sources have zero influence."""
    weights = matrix[None, :, :] * mask[:, None, :]
    mass = weights.sum(-1)
    result = jnp.einsum("bts,bsd->btd", weights, jnp.where(mask[..., None], values, 0))
    return result / jnp.maximum(mass[..., None], 1e-8), mass > 0
