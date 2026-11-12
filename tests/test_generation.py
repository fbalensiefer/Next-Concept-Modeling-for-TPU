from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm import Batch, NCM, RVQConfig, TransformerConfig, initialize
from ncm.generation import (PriorState, encode, evaluate_prior, generate,
                            level_geometry, make_prior_step)


def fixture():
    model = NCM(RVQConfig((1, 2, 4), 4, 4), TransformerConfig(8, 2, 1), 256)
    batch = Batch(jnp.array([[65, 66, 67, 0], [66, 65, 66, 65]]),
                  jnp.array([[1, 1, 1, 0], [1, 1, 1, 1]], bool),
                  jnp.zeros((2, 4), jnp.int32), jnp.zeros((2, 4), bool))
    codec = initialize(model, optax.adam(.001), jax.random.key(0), batch)
    return model, batch, codec


def test_prior_updates_only_prior_and_empty_batches_skip():
    model, batch, codec = fixture()
    snapshot = serialization.to_bytes(codec)
    params = codec.params['SharedLevelGenerator_0']
    codec_params = codec.params['MultiscaleTokenizer_0']
    optimizer = optax.adam(.01)
    state = PriorState(params, optimizer.init(params), jnp.array(0))
    target = encode(model, codec_params, codec.codebooks, batch)
    step = jax.jit(make_prior_step(model, optimizer, codec_params, codec.codebooks))
    updated, loss = step(state, batch, jax.random.key(2))
    assert jnp.isfinite(loss) and updated.step == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))
    assert serialization.to_bytes(codec) == snapshot
    np.testing.assert_array_equal(target.indices, encode(model, codec_params, codec.codebooks, batch).indices)
    empty, loss = step(updated, batch.replace(mask=jnp.zeros_like(batch.mask)), jax.random.key(3))
    for a, b in zip(jax.tree.leaves(updated), jax.tree.leaves(empty)):
        np.testing.assert_array_equal(a, b)
    assert loss == 0


def test_seeded_sampling_and_geometry():
    model, batch, codec = fixture()
    codec_params = codec.params['MultiscaleTokenizer_0']
    params = codec.params['SharedLevelGenerator_0']
    def sample(seed, temperature=5., top_k=0):
        return generate(model, codec_params, params, codec.codebooks, jax.random.key(seed),
                        batch_size=16, length=3, temperature=temperature, top_k=top_k)
    first = sample(1)
    repeated = sample(1)
    different = sample(2)
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[1], different[1])
    assert len(np.unique(np.asarray(first[0]), axis=0)) > 1
    assert np.all(np.asarray(first[0])[:, 3] == -1)
    greedy = sample(1, 0.)
    top_one = sample(2, 1., 1)
    np.testing.assert_array_equal(greedy[1], top_one[1])
    mask = jnp.broadcast_to(jnp.arange(4) < 3, (16, 4))
    geometry = level_geometry(model.rvq, mask)
    assert np.all(np.asarray(first[1])[~np.asarray(geometry)] == -1)


def test_exposure_metrics_first_level_and_codec_ceiling():
    model, batch, codec = fixture()
    stats = evaluate_prior(model, codec.params['MultiscaleTokenizer_0'],
                           codec.params['SharedLevelGenerator_0'], codec.codebooks, batch)
    np.testing.assert_allclose(stats['teacher_nll_sum'][0], stats['free_nll_sum'][0])
    np.testing.assert_array_equal(stats['count'], [2, 4, 7])
    assert stats['byte_count'] == 7
