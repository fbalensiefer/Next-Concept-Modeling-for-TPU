from dataclasses import replace

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm import Batch, CoarseConfig, CodebookState, NCM, RVQConfig, TransformerConfig, initialize, make_train_step
from ncm.coarse import (initialize_coarse, kmeans, maintain_coarse, new_coarse_state,
                        schedules)
from ncm.metrics import code_usage
from ncm.rvq import ema_update, init_codebooks, quantize
from ncm.scale_experiment import state_bytes


def test_masked_kmeans_is_distinct_deterministic_and_initializes_only_once():
    points = jnp.array([[-2.], [-2.], [2.], [2.], [9999.]])
    valid = jnp.array([1, 1, 1, 1, 0], bool)
    run = jax.jit(lambda key: kmeans(points, valid, 2, key, iterations=5))
    centers, counts = run(jax.random.key(4))
    repeat, _ = run(jax.random.key(4))
    np.testing.assert_array_equal(centers, repeat)
    np.testing.assert_allclose(np.sort(np.asarray(centers[:, 0])), [-2, 2], atol=.001)
    np.testing.assert_array_equal(counts, [2, 2])
    same, _ = kmeans(jnp.ones((3, 2)), jnp.ones(3, bool), 3, jax.random.key(1))
    assert len(np.unique(np.asarray(same), axis=0)) == 3
    cfg = RVQConfig((1, 2), 2, 1, coarse=CoarseConfig(enabled=True, kmeans_iterations=5))
    books = init_codebooks(jax.random.key(1), cfg)
    state = new_coarse_state(books, cfg.coarse, jax.random.key(2))
    latent = jnp.array([[[-2.], [-2.]], [[2.], [2.]]])
    fitted, state = initialize_coarse(books, state, latent, jnp.ones((2, 2), bool), cfg)
    assert state.initialized and state.initial_samples == 2
    np.testing.assert_array_equal(fitted.embeddings[1:], books.embeddings[1:])
    again, next_state = initialize_coarse(fitted, state, latent + 1000, jnp.ones((2, 2), bool), cfg)
    assert state_bytes((again, next_state)) == state_bytes((fitted, state))
    empty, uninitialized = initialize_coarse(books, new_coarse_state(books, cfg.coarse, jax.random.key(2)),
                                            latent, jnp.zeros((2, 2), bool), cfg)
    assert not uninitialized.initialized
    assert state_bytes(empty) == state_bytes(books)


def test_dead_replacement_uses_valid_batch_latents_and_restores_perplexity():
    settings = CoarseConfig(enabled=True, usage_decay=0., dead_code_threshold=.5,
                            dead_code_grace_steps=0, replacement_noise=.001)
    cfg = RVQConfig((1,), 2, 1, coarse=settings)
    embedding = jnp.array([[[0.], [100.]]])
    books = CodebookState(embedding, jnp.array([[1000., 1.]]), embedding * jnp.array([[[1000.], [1.]]]))
    state = new_coarse_state(books, settings, jax.random.key(0)).replace(initialized=jnp.array(True))
    # The invalid outlier must never become a replacement candidate.
    latent = jnp.array([[[10.]], [[10.]], [[99999.]]])
    mask = jnp.array([[1], [1], [0]], bool)
    output = quantize(latent, mask, books, cfg)
    step = jax.jit(lambda: maintain_coarse(books, state, output.counts, output.sums, latent, mask, cfg))
    updated, tracking, metrics = step()
    repeated, repeat_tracking, _ = step()
    assert state_bytes((updated, tracking)) == state_bytes((repeated, repeat_tracking))
    assert metrics['coarse_dead_replacements'] == 1
    np.testing.assert_array_equal(tracking.replacements, [0, 1])
    assert abs(float(updated.embeddings[0, 1, 0]) - 10) < .01
    assert tracking.hard_counts[1] >= settings.reset_count
    probe = jnp.array([[[0.]], [[10.]]])
    before = quantize(probe, jnp.ones((2, 1), bool), books, cfg)
    after = quantize(probe, jnp.ones((2, 1), bool), updated, cfg)
    assert code_usage(before.counts)['perplexity'][0] == 1
    np.testing.assert_allclose(code_usage(after.counts)['perplexity'][0], 2, atol=1e-6)
    np.testing.assert_allclose(updated.sums[0, 1] / (updated.counts[0, 1] + cfg.epsilon), updated.embeddings[0, 1])


def test_schedules_and_coarse_optimizer_group_leave_fine_updates_unchanged():
    settings = CoarseConfig(enabled=True, dead_code_threshold=0., temp_anneal_steps=10, beta_warmup_steps=10)
    cfg = RVQConfig((1, 1), 2, 1, decay=.5, coarse=settings)
    books = init_codebooks(jax.random.key(0), cfg)
    state = new_coarse_state(books, settings, jax.random.key(1)).replace(initialized=jnp.array(True))
    np.testing.assert_allclose(schedules(settings, state), [2., 0.])
    np.testing.assert_allclose(schedules(settings, state.replace(step=jnp.array(5))), [1.125, .125])
    np.testing.assert_allclose(schedules(settings, state.replace(step=jnp.array(20))), [.25, .25])
    counts, sums = jnp.ones((2, 2)) * 3, jnp.ones((2, 2, 1)) * 5
    latent, mask = jnp.ones((2, 1, 1)), jnp.ones((2, 1), bool)
    expected = ema_update(books, counts, sums, cfg)
    updated, _, _ = maintain_coarse(books, state, counts, sums, latent, mask, cfg)
    np.testing.assert_allclose(updated.embeddings[0], books.embeddings[0] + .25 * (expected.embeddings[0] - books.embeddings[0]), atol=1e-6)
    np.testing.assert_array_equal(updated.embeddings[1:], expected.embeddings[1:])
    np.testing.assert_array_equal(updated.counts, expected.counts)
    assert 'coarse_codebook' in state.optimizer_state.inner_states


def test_core_training_resumes_bit_exact_with_replacement_and_freezes_coarse():
    settings = CoarseConfig(enabled=True, warmup_batches=1, kmeans_iterations=3, usage_decay=0.,
                            dead_code_threshold=4., dead_code_grace_steps=0, temp_anneal_steps=3, beta_warmup_steps=3)
    cfg = RVQConfig((1, 2, 4), 2, 4, coarse=settings)
    model = NCM(cfg, TransformerConfig(8, 2, 1), 256)
    batch = Batch(jnp.array([[65, 66, 65, 66], [66, 65, 66, 65]]), jnp.ones((2, 4), bool),
                  jnp.zeros((2, 4), jnp.int32), jnp.zeros((2, 4), bool))
    optimizer = optax.adam(.001)
    initial = initialize(model, optimizer, jax.random.key(2), batch)
    assert initial.opt_state['coarse'].initialized
    step = jax.jit(make_train_step(model, optimizer))
    middle, metrics = step(initial, batch, jnp.array(0), jax.random.key(3))
    assert metrics['coarse_dead_replacements'] > 0
    expected, expected_metrics = step(middle, batch, jnp.array(0), jax.random.key(4))
    template = initialize(model, optimizer, jax.random.key(2), batch, initialize_codebooks=False)
    restored = serialization.from_bytes(template, state_bytes(middle))
    actual, actual_metrics = step(restored, batch, jnp.array(0), jax.random.key(4))
    assert state_bytes(actual) == state_bytes(expected)
    assert state_bytes(actual_metrics) == state_bytes(expected_metrics)
    assert actual.opt_state['coarse'].step == actual.step == 2
    assert actual.opt_state['coarse'].initial_samples == initial.opt_state['coarse'].initial_samples
    frozen, _ = step(actual, batch, jnp.array(1), jax.random.key(5))
    np.testing.assert_array_equal(actual.codebooks.embeddings[0], frozen.codebooks.embeddings[0])
    np.testing.assert_array_equal(actual.opt_state['coarse'].hard_counts, frozen.opt_state['coarse'].hard_counts)
    np.testing.assert_array_equal(actual.opt_state['coarse'].replacements, frozen.opt_state['coarse'].replacements)
    empty, _ = step(actual, batch.replace(mask=jnp.zeros_like(batch.mask)), jnp.array(0), jax.random.key(6))
    assert state_bytes(empty) == state_bytes(actual)
    recovered, _ = step(template, batch, jnp.array(0), jax.random.key(3))
    assert recovered.opt_state['coarse'].initialized


def test_legacy_optimizer_and_codebook_checkpoint_schema_is_preserved():
    cfg = RVQConfig((1, 2), 2, 2)
    model = NCM(cfg, TransformerConfig(4, 1, 1), 256)
    batch = Batch(jnp.array([[65, 66]]), jnp.ones((1, 2), bool), jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2), bool))
    state = initialize(model, optax.adam(.001), jax.random.key(0), batch)
    raw = serialization.msgpack_restore(state_bytes(state))
    assert set(raw) == {'params', 'opt_state', 'codebooks', 'step'}
    assert set(raw['codebooks']) == {'embeddings', 'counts', 'sums'}
    assert 'coarse' not in raw['opt_state']
    assert state_bytes(serialization.from_bytes(state, state_bytes(state))) == state_bytes(state)


def test_temperature_changes_soft_entropy_only_and_beta_weights_only_coarse_error():
    from ncm import LossConfig
    from ncm.losses import combined_loss, entropy_loss
    cfg = RVQConfig((1, 2), 2, 1)
    books = CodebookState(jnp.array([[[-1.], [1.]], [[-1.], [1.]]]), jnp.ones((2, 2)),
                          jnp.array([[[-1.], [1.]], [[-1.], [1.]]]))
    latent = jnp.array([[[.4], [.4]], [[.7], [.7]]])
    mask = jnp.ones((2, 2), bool)
    cold = quantize(latent, mask, books, cfg, coarse_temperature=.25)
    smooth = quantize(latent, mask, books, cfg, coarse_temperature=2.)
    np.testing.assert_array_equal(cold.indices, smooth.indices)
    np.testing.assert_allclose(cold.soft_counts[1:], smooth.soft_counts[1:])
    assert -entropy_loss(smooth.soft_counts[:1], jnp.ones(1)) > -entropy_loss(cold.soft_counts[:1], jnp.ones(1))
    logits = jnp.zeros((2, 2, 2))
    prior = jnp.zeros((*cold.indices.shape, 2))
    normal, _ = combined_loss(logits, jnp.zeros((2, 2), jnp.int32), mask, cold, prior, LossConfig())
    warm, _ = combined_loss(logits, jnp.zeros((2, 2), jnp.int32), mask, cold, prior, LossConfig(), coarse_beta=0.)
    expected = .25 * cold.commitment_numerator[0] / cold.commitment_denominator.sum()
    np.testing.assert_allclose(normal - warm, expected, atol=2e-6)
    grad = jax.grad(lambda x: entropy_loss(quantize(x, mask, books, cfg, coarse_temperature=2.).soft_counts[:1], jnp.ones(1)))(latent)
    assert jnp.isfinite(grad).all() and jnp.linalg.norm(grad) > 0
