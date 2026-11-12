import jax
import jax.numpy as jnp
import numpy as np

from ncm import RVQConfig, TransformerConfig
from ncm.autoregressive import AutoregressiveBytes, shifted_inputs, sample_ar
from ncm.fast_generation import generate_fast
from ncm.generation import generate
from ncm.grids import align, grid_matrices
from ncm.sparse_grids import grid_plan, sparse_align


def test_ar_cache_matches_full_prefix_and_is_causal():
    model = AutoregressiveBytes(TransformerConfig(8, 2, 2), 5)
    tokens = jnp.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    inputs = shifted_inputs(tokens)
    valid = jnp.ones_like(tokens, bool)
    params = model.init(jax.random.key(0), inputs, valid)['params']
    expected = model.apply({'params': params}, inputs, valid)
    changed = model.apply({'params': params}, inputs.at[:, 3:].set(17), valid)
    np.testing.assert_allclose(expected[:, :3], changed[:, :3], atol=1e-6)
    cache = tuple((jnp.zeros((2, 5, 2, 4)), jnp.zeros((2, 5, 2, 4))) for _ in range(2))
    actual = []
    for i in range(5):
        logits, cache = model.apply({'params': params}, inputs[:, i], jnp.array(i), cache,
                                   method=model.decode_step)
        actual.append(logits)
    np.testing.assert_allclose(expected, jnp.stack(actual, 1), atol=2e-6, rtol=2e-6)
    from flax import serialization
    params = serialization.msgpack_restore(serialization.to_bytes(params))
    first = sample_ar(model, params, jax.random.key(4), batch_size=2, length=5)
    repeat = sample_ar(model, params, jax.random.key(4), batch_size=2, length=5)
    np.testing.assert_array_equal(first, repeat)


def test_sparse_alignment_matches_dense_with_holes_and_nondivisible_grids():
    config = RVQConfig((1, 3, 5), 4, 2)
    down, up = grid_matrices(config)
    values = jnp.arange(20, dtype=jnp.float32).reshape(2, 5, 2)
    mask = jnp.array([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0]], bool)
    for i, count in enumerate(config.lengths):
        plan_down, plan_up = grid_plan(5, count)
        dense, dense_mask = align(values, mask, down[i, :count])
        sparse, sparse_mask = sparse_align(values, mask, plan_down)
        np.testing.assert_allclose(dense, sparse, atol=1e-6)
        np.testing.assert_array_equal(dense_mask, sparse_mask)
        a, am = align(dense, dense_mask, up[i, :, :count])
        b, bm = sparse_align(sparse, sparse_mask, plan_up)
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_array_equal(am, bm)


def test_fast_greedy_matches_reference_and_samples_reproducibly():
    from test_generation import fixture
    model, _, codec = fixture()
    args = (model, codec.params['MultiscaleTokenizer_0'], codec.params['SharedLevelGenerator_0'],
            codec.codebooks, jax.random.key(7))
    for length in (2, 4):
        expected = generate(*args, batch_size=3, length=length, temperature=0.)
        actual = jax.jit(lambda: generate_fast(*args, batch_size=3, length=length, temperature=0.))()
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
    a = generate_fast(*args, batch_size=8)
    b = generate_fast(*args, batch_size=8)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
