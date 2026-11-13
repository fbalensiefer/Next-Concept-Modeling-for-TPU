import numpy as np
import pytest
from ncm.representations import encode_pairs, decode_pairs


def test_all_byte_pairs_are_bijective():
    codes = np.arange(65536, dtype=np.int32).reshape(-1, 1)
    restored = decode_pairs(codes)
    np.testing.assert_array_equal(encode_pairs(restored), codes)
    assert len(np.unique(restored, axis=0)) == 65536


@pytest.mark.parametrize('values', [np.array([[256, 0]]), np.array([[-1, 0]]),
                                    np.array([[0.5, 1.0]]), np.array([[1]])])
def test_invalid_pair_inputs_rejected(values):
    with pytest.raises(ValueError):
        encode_pairs(values)
