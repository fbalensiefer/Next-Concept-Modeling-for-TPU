import jax
import jax.numpy as jnp
import numpy as np

from ncm.losses import concept_consistency, cross_entropy, entropy_loss, next_token_cross_entropy
from ncm.metrics import exact_match


def test_empty_and_padding_safe_losses():
    logits = jnp.zeros((1, 3, 4))
    targets = jnp.array([[0, 1, -1]])
    mask = targets >= 0
    np.testing.assert_allclose(cross_entropy(logits, targets, mask), np.log(4))
    np.testing.assert_allclose(next_token_cross_entropy(logits, targets, mask), np.log(4))
    assert cross_entropy(logits, targets, jnp.zeros_like(mask)) == 0
    features = jnp.ones((1, 3, 2))
    np.testing.assert_allclose(concept_consistency(features, features, mask), 0, atol=1e-6)
    assert concept_consistency(features, features, jnp.zeros_like(mask)) == 0


def test_entropy_has_gradient_and_rewards_diversity():
    weights = jnp.array([2., 1.])
    uniform = jnp.ones((2, 4))
    collapsed = jnp.array([[4., 0, 0, 0], [4., 0, 0, 0]])
    assert entropy_loss(uniform, weights) < entropy_loss(collapsed, weights)
    counts = jnp.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    grad = jax.grad(entropy_loss)(counts, weights)
    assert jnp.all(jnp.isfinite(grad)) and jnp.linalg.norm(grad) > 0
    assert entropy_loss(uniform * 0, weights) == 0


def test_exact_match_excludes_empty_examples():
    mask = jnp.array([[True, True], [False, False]])
    metrics = exact_match(jnp.array([[1, 0], [9, 9]]), jnp.array([[1, 2], [0, 0]]), mask)
    assert metrics["token_exact_match"] == .5
    assert metrics["sequence_exact_match"] == 0
