"""Preregistered power study. Test-only corrupt verifiers never enter src/."""
from functools import lru_cache
import itertools
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ncm.speculative import DistributionAdapter, speculative_generate

N = 200_000
D, L, V = 3, 4, 3
T = jnp.array([[.55, .35, .10], [.15, .60, .25], [.30, .15, .55]])


def teacher(_, buffer, position, length):
    previous = jax.lax.dynamic_slice(buffer, (position-1,), (length+1,))
    return jnp.log(T[previous])


def proposal(_, buffer, position, length, key):
    z = jax.random.bernoulli(key).astype(jnp.int32)
    base = jnp.array([[.82, .13, .05], [.08, .18, .74]])[z]
    rows = jnp.arange(length)[:, None]
    indices = (jnp.arange(V)[None] + rows + position) % V
    return jnp.log(base[indices])


def write_result(name, **values):
    path = Path('reports/validation-controls/power.json')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = dict(git_commit=commit, **values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


@lru_cache(None)
def reference():
    words = np.array(list(itertools.product(range(V), repeat=L)))
    law = np.ones(len(words))
    previous = np.zeros(len(words), np.int32)
    for j in range(L):
        law *= np.asarray(T, np.float64)[previous, words[:, j]]
        previous = words[:, j]
    law /= law.sum()
    rng = np.random.default_rng(201)
    floors = np.array([np.abs(rng.multinomial(N, law)/N-law).sum()/2 for _ in range(20)])
    return law, floors, float(np.percentile(floors, 95))


def statistic(tokens):
    law, floors, threshold = reference()
    indices = np.asarray(tokens) @ (V**np.arange(L-1, -1, -1))
    empirical = np.bincount(indices, minlength=len(law))/len(indices)
    return float(np.abs(empirical-law).sum()/2), threshold, floors


def corrupt(key, variant):
    """Small test-only implementation with one controlled error per variant."""
    prompt_len = 1
    size = prompt_len+L+D+1
    buffer = jnp.zeros(size, jnp.int32)
    def condition(state):
        return state[1] < L+prompt_len
    def step(state):
        buffer, position, key = state
        key, latent, token_key, accept_key, bonus_key = jax.random.split(key, 5)
        q = jax.nn.softmax(proposal(None, buffer, position, D, latent))
        tokens = jax.random.categorical(token_key, jnp.log(q)).astype(jnp.int32)
        reported = q
        if variant == 'mismatched_q':
            reported = jax.nn.softmax(jnp.log(q)/1.2)
        if variant == 'future_leak':
            future = jnp.concatenate([tokens[1:], tokens[-1:]])
            reported = jax.nn.softmax(jnp.log(q)+2*jax.nn.one_hot(future, V))
        proposed = jax.lax.dynamic_update_slice(buffer, tokens, (position,))
        p = jax.nn.softmax(teacher(None, proposed, position, D))
        selected = jnp.arange(D)
        ratios = p[selected, tokens]/reported[selected, tokens]
        accepted = jax.random.uniform(accept_key, (D,)) < jnp.minimum(1., ratios)
        prefix = jnp.cumprod(accepted.astype(jnp.int32)).sum()
        index = jnp.minimum(prefix, D-1)
        residual = jnp.maximum(p[index]-reported[index], 0)
        if variant == 'bad_residual':
            residual = jnp.maximum(residual, .05)
        residual /= jnp.maximum(residual.sum(), 1e-30)
        bonus_p = jnp.where(prefix == D, reported[-1] if variant == 'wrong_bonus' else p[-1], residual)
        bonus = jax.random.categorical(bonus_key, jnp.log(bonus_p)).astype(jnp.int32)
        keep = (jnp.arange(size) >= position) & (jnp.arange(size) < position+prefix)
        buffer = jnp.where(keep, proposed, buffer).at[position+prefix].set(bonus)
        return buffer, position+prefix+1, key
    buffer, _, _ = jax.lax.while_loop(condition, step, (buffer, jnp.asarray(1), key))
    return buffer[1:L+1]


def test_large_sample_exactness():
    keys = jax.random.split(jax.random.key(200), N)
    tokens = jax.jit(jax.vmap(lambda key: speculative_generate(
        DistributionAdapter(None, proposal), DistributionAdapter(None, teacher),
        jnp.array([0]), D, L, key).tokens))(keys)
    tv, threshold, floors = statistic(tokens)
    write_result('exact', seed=200, samples=N, support=V**L, tv=tv,
                 floor_95=threshold, direct_floors=floors.tolist(),
                 conservative_scale=float(np.sqrt(V**L/(2*N))), passed=tv <= threshold)
    assert tv <= threshold


@pytest.mark.parametrize('variant', ['mismatched_q','bad_residual','future_leak','wrong_bonus'])
def test_negative_control(variant):
    keys = jax.random.split(jax.random.key(200), N)
    tokens = jax.jit(jax.vmap(lambda key: corrupt(key, variant)))(keys)
    tv, threshold, floors = statistic(tokens)
    write_result(variant, seed=200, samples=N, tv=tv, floor_95=threshold, rejected=tv > threshold)
    assert tv > threshold, f'power failure: {variant} TV={tv} floor={threshold}'


def test_greedy_equality():
    def greedy_teacher(_, buffer, position, length):
        prev = jax.lax.dynamic_slice(buffer, (position-1,), (length+1,))
        target = (prev*2+1) % 3
        return jnp.where(jax.nn.one_hot(target, 3), 0., -jnp.inf)
    def greedy_draft(_, buffer, position, length, key):
        target = (jnp.arange(length)+buffer[position-1]) % 3
        return jnp.where(jax.nn.one_hot(target, 3), 0., -jnp.inf)
    prompts = jax.random.randint(jax.random.key(202), (50, 3), 0, 3)
    def direct(row):
        def step(prev, _):
            token = (prev*2+1) % 3
            return token, token
        return jax.lax.scan(step, row[-1], None, length=11)[1]
    expected = jax.vmap(direct)(prompts)
    for length in (1,2,3,4,7):
        actual = jax.jit(jax.vmap(lambda row: speculative_generate(
            DistributionAdapter(None, greedy_draft), DistributionAdapter(None, greedy_teacher),
            row, length, 11, jax.random.key(202)).tokens))(prompts)
        np.testing.assert_array_equal(actual, expected)
    write_result('greedy', seed=202, prompts=50, draft_lengths=[1,2,3,4,7], passed=True)


@pytest.mark.parametrize('prefix', [0,1,3,4])
@pytest.mark.parametrize('length', [1,3,5,9])
def test_bookkeeping(prefix, length):
    def draft(_, buffer, position, d, key):
        return jnp.broadcast_to(jnp.array([0., -jnp.inf]), (d,2))
    def teacher(_, buffer, position, d):
        # Each new round accepts exactly prefix positions, then emits 1.
        token = (jnp.arange(d+1) == prefix).astype(jnp.int32)
        return jnp.where(jax.nn.one_hot(token, 2), 0., -jnp.inf)
    output = speculative_generate(DistributionAdapter(None,draft), DistributionAdapter(None,teacher),
                                  jnp.array([0]),4,length,jax.random.key(200))
    committed = prefix+1
    passes = (length+committed-1)//committed
    expected = np.tile(np.r_[np.zeros(prefix,np.int32),1], passes)[:length]
    np.testing.assert_array_equal(output.tokens, expected)
    remaining = [length-i*committed for i in range(passes)]
    accepted = sum(min(prefix,r) for r in remaining)
    proposed = sum(min(4,r) for r in remaining)
    assert int(output.teacher_forward_passes)==passes
    np.testing.assert_allclose(output.tokens_per_forward_pass,length/passes)
    np.testing.assert_allclose(output.acceptance_rate,accepted/proposed)
