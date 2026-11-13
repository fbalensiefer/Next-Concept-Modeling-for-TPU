"""Exact byte and fixed-pair controls; these are representations, not priors."""
import numpy as np


def _integers(values, maximum):
    values = np.asarray(values)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Expected an integer [batch,length] array')
    if np.any(values < 0) or np.any(values > maximum):
        raise ValueError('Values outside representation range')
    return values.astype(np.int32)


def encode_pairs(tokens):
    """Pack adjacent bytes big-endian into categories 0..65535, losslessly."""
    tokens = _integers(tokens, 255)
    if tokens.shape[1] % 2:
        raise ValueError('Fixed pairs require an even byte length')
    return tokens[:, ::2] * 256 + tokens[:, 1::2]


def decode_pairs(codes):
    codes = _integers(codes, 65535)
    return np.stack((codes // 256, codes % 256), axis=-1).reshape(len(codes), -1)
