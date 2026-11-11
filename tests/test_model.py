import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm import (Batch, LossConfig, MultiscaleTokenizer, NCM, RVQConfig,
                 SharedLevelGenerator, TransformerConfig, init_codebooks,
                 initialize, make_train_step)
from ncm.sharding import compile_train_step, create_mesh


def fixture():
    config = RVQConfig((1, 2, 4), 4, 4)
    batch = Batch(jnp.array([[1, 2, 3, 0], [2, 3, 1, 0]]),
                  jnp.array([[True, True, True, False]] * 2),
                  jnp.array([[0, 65, 0, 0]] * 2),
                  jnp.array([[False, True, False, False]] * 2))
    return config, batch


def test_generator_no_current_or_future_target_leakage():
    config, _ = fixture()
    generator = SharedLevelGenerator(config, TransformerConfig(8, 2, 1))
    books = init_codebooks(jax.random.key(0), config)
    mask = jnp.broadcast_to(jnp.arange(4)[None, None, :] < jnp.array(config.lengths)[:, None, None], (3, 2, 4))
    targets = jnp.zeros(mask.shape, jnp.int32)
    params = generator.init(jax.random.key(1), books, mask, targets)
    apply = jax.jit(lambda t, tf: generator.apply(params, books, mask, t, teacher_forcing=tf))
    first = apply(targets, jnp.array(True))
    # Change every code at level 1 and later: logits through level 1 are identical.
    changed = apply(targets.at[1:].set(1), jnp.array(True))
    np.testing.assert_array_equal(first.logits[:2], changed.logits[:2])
    assert not np.allclose(first.logits[2], changed.logits[2])
    free_a = apply(targets, jnp.array(False))
    free_b = apply(targets + 2, jnp.array(False))
    np.testing.assert_array_equal(free_a.logits, free_b.logits)
    assert np.all(np.asarray(free_a.indices)[~np.asarray(mask)] == -1)


def test_byte_fallback_excludes_codes_and_routes_embeddings():
    config, batch = fixture()
    books = init_codebooks(jax.random.key(0), config)
    tokenizer = MultiscaleTokenizer(config, 256)
    args = (batch.tokens, batch.mask, batch.fallback_bytes, batch.fallback_mask, books)
    params = tokenizer.init(jax.random.key(1), *args)
    result = tokenizer.apply(params, *args)
    changed_tokens = batch.tokens.at[:, 1].set(123).at[:, 3].set(201)
    changed = tokenizer.apply(params, changed_tokens, *args[1:])
    np.testing.assert_array_equal(result.rvq.indices, changed.rvq.indices)
    assert not result.rvq.mask[-1, :, 1].any()
    table = params["params"]["byte_embedding"]["embedding"]
    np.testing.assert_allclose(result.representation[:, 1], jnp.broadcast_to(table[65], (2, 4)))
    decoded = tokenizer.apply(params, result.rvq.quantized, batch.mask,
                              batch.fallback_bytes, batch.fallback_mask,
                              method=tokenizer.decode)
    np.testing.assert_allclose(decoded, result.logits)


def test_compiled_training_phase_and_empty_batch():
    config, batch = fixture()
    model = NCM(config, TransformerConfig(8, 2, 1), 256, LossConfig(concept=.1))
    optimizer = optax.adamw(.001)
    state = initialize(model, optimizer, jax.random.key(0), batch)
    step, state, batch = compile_train_step(make_train_step(model, optimizer), state,
                                           batch, create_mesh(1, 1, jax.devices()[:1]))
    new_state, metrics = step(state, batch, jnp.array(1), jax.random.key(2))
    assert new_state.step == 1
    assert all(jnp.isfinite(v).all() for v in metrics.values())
    assert metrics["grad_norm"] > 0
    np.testing.assert_array_equal(state.codebooks.embeddings[0], new_state.codebooks.embeddings[0])
    assert not np.array_equal(state.codebooks.embeddings[-1], new_state.codebooks.embeddings[-1])
    empty = batch.replace(mask=jnp.zeros_like(batch.mask))
    skipped, metrics = step(new_state, empty, jnp.array(0), jax.random.key(3))
    for before, after in zip(jax.tree.leaves(new_state), jax.tree.leaves(skipped)):
        np.testing.assert_array_equal(before, after)
    assert metrics["loss"] == 0
