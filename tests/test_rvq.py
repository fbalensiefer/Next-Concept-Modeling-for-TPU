import jax
import jax.numpy as jnp
import numpy as np

from ncm import CodebookState, RVQConfig, ema_update, init_codebooks, quantize
from ncm.grids import align, grid_matrices
from ncm.losses import entropy_loss


def test_residual_nearest_neighbor_and_statistics():
    config = RVQConfig((2, 2), 2, 1, decay=.5)
    embeddings = jnp.array([[[0.], [2.]], [[0.], [1.]]])
    state = CodebookState(embeddings, jnp.ones((2, 2)), embeddings)
    output = jax.jit(lambda x: quantize(x, jnp.array([[True, False]]), state, config))(
        jnp.array([[[2.8], [900.]]]))
    np.testing.assert_array_equal(output.indices, [[[1, -1]], [[1, -1]]])
    np.testing.assert_allclose(output.quantized, [[[3.], [0.]]])
    np.testing.assert_allclose(output.counts, [[0, 1], [0, 1]])
    np.testing.assert_allclose(output.sums[:, 1, 0], [2.8, .8], atol=1e-6)


def test_ema_equations_and_runtime_phase():
    config = RVQConfig((1, 2, 4, 8), 3, 2, decay=.5)
    state = init_codebooks(jax.random.key(0), config)
    counts, sums = jnp.ones_like(state.counts) * 3, jnp.ones_like(state.sums) * 5
    update = jax.jit(lambda phase: ema_update(state, counts, sums, config, phase=phase))
    joint, final = update(jnp.array(0)), update(jnp.array(1))
    np.testing.assert_allclose(joint.counts, 2)
    np.testing.assert_allclose(joint.sums, .5 * state.sums + 2.5)
    np.testing.assert_allclose(joint.embeddings, joint.sums / (2 + config.epsilon))
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(final)):
        np.testing.assert_array_equal(before[:2], after[:2])
    np.testing.assert_array_equal(final.embeddings[2:], joint.embeddings[2:])
    empty = ema_update(state, counts * 0, sums * 0, config)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(empty)):
        np.testing.assert_array_equal(before, after)


def test_multiscale_padding_and_straight_through_gradient():
    config = RVQConfig((2, 3, 5), 4, 2)
    state = init_codebooks(jax.random.key(1), config)
    mask = jnp.array([[True, True, True, False, False]])
    x = jnp.arange(10, dtype=jnp.float32).reshape(1, 5, 2)
    fn = lambda x: quantize(x, mask, state, config)
    a, b = fn(x), fn(jnp.where(mask[..., None], x, 9999))
    for first, second in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(jax.grad(lambda x: fn(x).quantized.sum())(x),
                               jnp.broadcast_to(mask[..., None], x.shape))
    entropy_grad = jax.grad(lambda x: entropy_loss(fn(x).soft_counts, jnp.ones(3)))(x)
    assert jnp.isfinite(entropy_grad).all() and jnp.linalg.norm(entropy_grad) > 0
    assert not np.any(a.mask[0, :, 2:])
    down, _ = grid_matrices(config)
    resampled, _ = align(jnp.ones_like(x), mask, down[1])
    np.testing.assert_allclose(resampled[0, :2], 1)


def test_final_phase_disables_nested_dropout():
    config = RVQConfig((1, 2, 4, 8), 4, 2, nested_dropout=1.)
    state = init_codebooks(jax.random.key(0), config)
    fn = jax.jit(lambda phase: quantize(
        jnp.ones((32, 8, 2)), jnp.ones((32, 8), bool), state, config,
        phase=phase, training=True, key=jax.random.key(3)))
    joint, final = fn(jnp.array(0)), fn(jnp.array(1))
    assert joint.mask.sum() < final.mask.sum()
    assert final.mask.sum() == 32 * sum(config.lengths)
