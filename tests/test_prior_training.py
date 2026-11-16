import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ncm.autoregressive import AutoregressiveCategorical
from ncm.config import PriorTrainingConfig, TransformerConfig
from ncm.prior_training import (PriorTrainState, flatten_rvq_codes,
                                make_categorical_prior_train_step,
                                shifted_categorical_inputs)


def test_flatten_rvq_codes_preserves_level_order_and_masks_padding():
    indices = jnp.asarray([[[0, 1, -1]], [[2, -1, -1]]])
    mask = indices >= 0
    codes, valid = flatten_rvq_codes(indices, mask, (2, 1))
    np.testing.assert_array_equal(codes, [[0, 1, 2]])
    np.testing.assert_array_equal(valid, [[True, True, True]])


def test_categorical_prior_loss_updates_prior_but_codec_is_outside_optimizer():
    config = TransformerConfig(width=8, heads=2, layers=1)
    prior = AutoregressiveCategorical(config, max_length=4, vocab_size=3)
    targets = jnp.asarray([[0, 1, 2, 1], [2, 0, 1, 0]], jnp.int32)
    valid = jnp.asarray([[True, True, True, False], [True, True, False, False]])
    params = prior.init(jax.random.key(0), shifted_categorical_inputs(targets, 3), valid)['params']
    optimizer = optax.adam(1e-2)
    state = PriorTrainState(params, optimizer.init(params), jnp.asarray(0))
    codec_weights = {'encoder': jnp.arange(6, dtype=jnp.float32)}
    codec_before = jax.tree.map(lambda x: np.array(x, copy=True), codec_weights)

    step = make_categorical_prior_train_step(prior, optimizer, 0.7)
    next_state, metrics = step(state, targets, valid, 1.25, 0.5)

    assert np.isfinite(float(metrics['prior_loss']))
    assert np.isfinite(float(metrics['total_loss']))
    assert float(metrics['total_loss']) == pytest.approx(
        1.25 + 0.5 + 0.7 * float(metrics['prior_loss']))
    assert int(next_state.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(state.params), jax.tree.leaves(next_state.params)))
    for actual, expected in zip(jax.tree.leaves(codec_weights), jax.tree.leaves(codec_before)):
        np.testing.assert_array_equal(actual, expected)


def test_categorical_prior_is_causal():
    config = TransformerConfig(width=8, heads=2, layers=1)
    prior = AutoregressiveCategorical(config, max_length=4, vocab_size=3)
    first = jnp.asarray([[3, 0, 1, 2]], jnp.int32)
    changed_future = first.at[0, 2:].set(jnp.asarray([2, 0]))
    valid = jnp.ones_like(first, bool)
    params = prior.init(jax.random.key(5), first, valid)['params']
    left = prior.apply({'params': params}, first, valid)
    right = prior.apply({'params': params}, changed_future, valid)
    np.testing.assert_allclose(left[:, :2], right[:, :2], rtol=1e-6, atol=1e-6)


def test_prior_config_requires_positive_weight_and_boolean_freeze_flag():
    with pytest.raises(ValueError):
        PriorTrainingConfig(prior_loss_weight=0.)
    with pytest.raises(ValueError):
        PriorTrainingConfig(freeze_codec=1)
