import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ncm.config import NCMConfig, RVQConfig, TransformerConfig
from ncm.generator import SharedLevelGenerator
from ncm.rvq import init_codebooks
from ncm.speculative import DistributionAdapter, speculative_generate


def fixture():
    rvq = RVQConfig((3, 6), 4, 4)
    generator = SharedLevelGenerator(rvq, TransformerConfig(8, 2, 1), NCMConfig(1, 2))
    books = init_codebooks(jax.random.key(0), rvq)
    masks = jnp.ones((2, 4, 6), bool)
    targets = jnp.zeros((2, 4, 6), jnp.int32)
    prompt = jnp.ones((4, 2), jnp.int32)
    params = generator.init(jax.random.key(1), books, masks, targets, prompt_ids=prompt)['params']
    return generator, books, masks, targets, prompt, params


def test_causal_and_prompt_invariance():
    gen, books, masks, targets, prompt, params = fixture()
    def apply(t, u):
        return gen.apply({'params': params}, books, masks, t, prompt_ids=u).logits
    original = apply(targets, prompt)
    changed = apply(targets.at[0, :, 1:].set(2), prompt)
    np.testing.assert_allclose(original[0, :, :2], changed[0, :, :2], atol=1e-6)
    assert not np.allclose(original[0, :, 2], changed[0, :, 2])
    assert not np.allclose(original[0], apply(targets, prompt*2)[0])
    free = jax.jit(lambda t: gen.apply({'params': params}, books, masks, t,
                       prompt_ids=prompt, teacher_forcing=False).indices)(targets)
    np.testing.assert_array_equal(free, gen.apply({'params': params}, books, masks,
        targets+2, prompt_ids=prompt, teacher_forcing=False).indices)


def test_speculative_joint_distribution():
    # History-dependent teacher, deliberately poor proposal, including zero mass.
    transition = jnp.array([[.1, .7, .2], [.5, .1, .4], [.3, .3, .4]])
    def draft(q, buffer, position, length, key):
        return jnp.broadcast_to(jnp.log(q), (length, 3))
    def teacher(p, buffer, position, length):
        previous = jax.lax.dynamic_slice(buffer, (position-1,), (length+1,))
        return jnp.log(p[previous])
    q = DistributionAdapter(jnp.array([.8, .2, 0.]), draft)
    p = DistributionAdapter(transition, teacher)
    keys = jax.random.split(jax.random.key(44), 16000)
    sample = jax.jit(jax.vmap(lambda key: speculative_generate(q, p, jnp.array([0]), 2, 2, key)))
    output = sample(keys)
    joint = np.asarray(output.tokens)
    observed = np.bincount(joint[:, 0]*3 + joint[:, 1], minlength=9)/len(keys)
    exact = (np.asarray(transition[0])[:, None]*np.asarray(transition)).ravel()
    assert np.abs(observed-exact).sum()/2 < .025
    def direct(key):
        a, b = jax.random.split(key)
        first = jax.random.categorical(a, jnp.log(transition[0]))
        second = jax.random.categorical(b, jnp.log(transition[first]))
        return first*3+second
    direct_hist = np.bincount(np.asarray(jax.jit(jax.vmap(direct))(keys)), minlength=9)/len(keys)
    assert np.abs(observed-direct_hist).sum()/2 < .03
    assert np.all(np.asarray(output.tokens_per_forward_pass) >= 1)


def test_full_acceptance_and_truncation():
    def logits(_, buffer, position, length, *rng):
        return jnp.broadcast_to(jnp.array([0., -jnp.inf]), (length + (not rng), 2))
    result = jax.jit(lambda k: speculative_generate(DistributionAdapter(None, logits),
        DistributionAdapter(None, logits), jnp.array([], jnp.int32), 3, 7, k))(jax.random.key(0))
    np.testing.assert_array_equal(result.tokens, np.zeros(7))
    assert float(result.acceptance_rate) == 1
    assert int(result.teacher_forward_passes) == 2


def test_gradient_flow_and_sharding():
    from ncm.sharding import create_mesh, parameter_shardings
    from jax.sharding import NamedSharding, PartitionSpec as P
    gen, books, masks, targets, prompt, params = fixture()
    def loss(params, targets, prompt):
        logits = gen.apply({'params': params}, books, masks, targets, prompt_ids=prompt).logits
        return jnp.square(logits).mean()
    value, grads = jax.jit(jax.value_and_grad(loss))(params, targets, prompt)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads['HybridGenerator_0']['code_embedding']['embedding'])) > 0
    if len(jax.devices()) < 4:
        pytest.skip('four devices required for dp/fsdp comparison')
    mesh = create_mesh(2, 2, jax.devices()[:4])
    specs = parameter_shardings(params, mesh)
    compiled = jax.jit(jax.value_and_grad(loss), in_shardings=(specs,
        NamedSharding(mesh, P(None, 'dp', None)), NamedSharding(mesh, P('dp', None))))
    actual, sharded_grads = compiled(jax.device_put(params, specs), targets, prompt)
    np.testing.assert_allclose(actual, value, atol=1e-5)
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(sharded_grads)):
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-4)


def test_data_hash_padding(tmp_path):
    from ncm.data import load_text
    path = tmp_path/'text.txt'
    path.write_text('hello café')
    data = load_text(path, 8)
    assert data.tokens.shape == (2, 8)
    assert data.mask.sum() == len('hello café'.encode())
    assert data.byte_counts.sum() == data.mask.sum()
    load_text(path, 8, expected_sha256=data.sha256)
    with pytest.raises(ValueError, match='SHA256'):
        load_text(path, 8, expected_sha256='bad')


def test_masked_codes_and_prompt_positions():
    gen, books, masks, targets, prompt, params = fixture()
    masks = masks.at[0, :, 1].set(False)
    before = gen.apply({'params': params}, books, masks, targets, prompt_ids=prompt).logits
    after = gen.apply({'params': params}, books, masks, targets.at[0, :, 1].set(3), prompt_ids=prompt).logits
    np.testing.assert_allclose(before[0, :, 2], after[0, :, 2], atol=1e-6)
    # Prompts are sequences, including when all code grids are parallel.
    pure = SharedLevelGenerator(gen.rvq, gen.transformer, NCMConfig(0, 2))
    ordered = prompt.at[:, 1].set(2)
    variables = pure.init(jax.random.key(19), books, masks, targets, prompt_ids=ordered)
    a = pure.apply(variables, books, masks, targets, prompt_ids=ordered).logits
    b = pure.apply(variables, books, masks, targets, prompt_ids=ordered[:, ::-1]).logits
    assert not np.allclose(a, b, atol=1e-7)
    pm = jnp.ones_like(ordered, bool).at[:, 1].set(False)
    a = pure.apply(variables, books, masks, targets, prompt_ids=ordered, prompt_mask=pm).logits
    b = pure.apply(variables, books, masks, targets, prompt_ids=ordered.at[:, 1].set(3), prompt_mask=pm).logits
    np.testing.assert_allclose(a, b, atol=1e-6)
