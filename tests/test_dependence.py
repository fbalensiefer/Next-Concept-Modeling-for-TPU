import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ncm.dependence import (codewords, conditional_tc, direct_factorization_kl,
                           hard_joint, soft_joint)
from ncm.scale_experiment import distribution_metrics, fit_canonical


def test_exact_dependence_calibration_and_explaining_shared_bit():
    # A pair carrying the same shared bit has one bit of total correlation.
    correlated = jnp.array([.5, 0., 0., .5])
    np.testing.assert_allclose(conditional_tc(correlated, (2,), 2), [np.log(2)], atol=1e-6)
    assert np.isclose(direct_factorization_kl(correlated, (2,), 2), np.log(2))
    independent = jnp.ones(4) / 4
    np.testing.assert_allclose(conditional_tc(independent, (2,), 2), 0, atol=1e-6)
    # Put the shared bit into a preceding level: conditional dependence is zero.
    joint = jnp.zeros(8).at[0].set(.5).at[7].set(.5)
    np.testing.assert_allclose(conditional_tc(joint, (1, 2), 2), 0, atol=1e-6)
    assert abs(direct_factorization_kl(joint, (1, 2), 2)) < 1e-8


def test_soft_hard_agreement_gradient_and_enumeration_limit():
    codes = jnp.array([[0, 0, 1], [1, 1, 0], [1, 0, 1]])
    weights = jnp.array([.2, .3, .5])
    probabilities = jax.nn.one_hot(codes, 2)
    hard = hard_joint(codes, weights, (1, 2), 2)
    soft = soft_joint(probabilities, weights, (1, 2), 2)
    np.testing.assert_allclose(hard, soft, atol=1e-7)
    logits = jnp.array([[[2., -1.], [1., -2.]], [[-1., 2.], [-2., 1.]]])
    def objective(x):
        joint = soft_joint(jax.nn.softmax(x), jnp.ones(2), (2,), 2)
        return conditional_tc(joint, (2,), 2).sum()
    grad = jax.grad(objective)(logits)
    assert jnp.isfinite(grad).all() and jnp.linalg.norm(grad) > 0
    with pytest.raises(ValueError, match='budget'):
        codewords((20,), 2)


def test_output_support_failure_is_not_hidden_by_smoothing():
    metrics = distribution_metrics(np.array([.5, .5]), np.array([1., 0., 0.]))
    assert metrics['total_variation'] == .5
    assert metrics['zero_support_teacher_mass'] == .5
    assert metrics['forward_kl_nats'] is None
    assert np.isfinite(metrics['jensen_shannon_nats'])


def test_canonical_projection_is_fixed_from_calibration():
    features = np.random.default_rng(0).normal(size=(8, 3, 4)).astype(np.float32)
    mean, projection, scale = fit_canonical(features, 2)
    assert projection.shape == (4, 2)
    np.testing.assert_allclose(projection.T @ projection, np.eye(2), atol=1e-6)
    assert np.all(scale > 0)
    np.testing.assert_allclose((((features - mean) @ projection) / scale).mean(axis=(0, 1)), 0, atol=1e-6)


def test_teacher_zero_noise_has_finite_source_kl(tmp_path):
    import json
    from pathlib import Path
    from ncm.scale_experiment import teacher_setup
    cfg = json.loads(Path('examples/scale/config.json').read_text())
    cfg.update(lengths=[1, 2], dim=4, teacher_width=8, teacher_steps=1,
               batch_size=4, checkpoint_every=1, source_noise=0.,
               train_samples=8, validation_samples=8, calibration_samples=8)
    _, probability, _, _, info = teacher_setup(cfg, tmp_path, False)
    np.testing.assert_allclose(probability.sum(), 1., atol=1e-7)
    assert np.isfinite(info['source_to_teacher_kl'])
