import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm.coverage import make_decoder_repair_step
from ncm.generation import PriorState
from ncm.scale_experiment import arm_weights, digest, run, state_bytes
from ncm.tokenizer import TokenDecoder


def test_four_arm_coefficients_are_a_factorial_control():
    config = {'alignment_weight': .05, 'dependency_weight': .25}
    expected = {'reconstruction': [0., 0.], 'alignment': [.05, 0.],
                'dependency': [0., .25], 'alignment_dependency': [.05, .25]}
    for arm, values in expected.items():
        np.testing.assert_allclose(arm_weights(arm, config), values)


def test_decoder_replay_reduces_training_error():
    decoder = TokenDecoder(4, 256, 2)
    values = jnp.array([[[1., 0., 0., 0.]] * 2, [[0., 1., 0., 0.]] * 2])
    tokens = jnp.array([[65, 65], [66, 66]])
    params = decoder.init(jax.random.key(0), values, jnp.ones((2, 2), bool))['params']
    optimizer = optax.adam(.01)
    state = PriorState(params, optimizer.init(params), jnp.array(0))
    step = jax.jit(make_decoder_repair_step(4, optimizer))
    state, first_loss = step(state, values, tokens)
    for _ in range(49):
        state, loss = step(state, values, tokens)
    assert loss < first_loss * .1
    assert state.step == 50


def test_followup_repair_resume_preserves_frozen_state_and_exact_outputs(tmp_path, monkeypatch):
    cfg = json.loads(Path('examples/scale/followup.json').read_text())
    cfg.update(codec_sampling='balanced_observed', lengths=[1, 2], dim=4, width=8, teacher_width=8, seeds=[3], confirmation_seeds=[3],
               min_alignment_winning_seeds=1, teacher_steps=2, codec_steps=2, prior_steps=2,
               batch_size=4, checkpoint_every=1, train_samples=16, validation_samples=16,
               calibration_samples=16, decoder_repair_steps=3)
    snapshots = []
    original = Path.replace
    def capture(path, target):
        result = original(path, target)
        if str(target).endswith('dependency/decoder_repair.msgpack') and not snapshots:
            snapshots.append(Path(target).read_bytes())
        return result
    monkeypatch.setattr(Path, 'replace', capture)
    output = tmp_path / 'followup'
    report = run(cfg, output)
    assert len(report['results']) == 4
    assert report['decision']['matched_student_flop_estimates']
    assert report['followup_decision']['confirmation_seeds'] == [3]
    for row in report['results']:
        assert row['decoder_repair']['encoder_and_codebooks_unchanged']
        assert row['decoder_repair']['steps'] == 3
        assert row['decoder_repair']['distinct_training_sequences'] <= 4
        np.testing.assert_allclose(row['before_repair']['hard_conditional_tc_sum'], row['hard_conditional_tc_sum'], atol=1e-7)
        directory = output / 'seed-3' / row['arm']
        base = serialization.msgpack_restore((directory / 'codec.msgpack').read_bytes())
        repaired = serialization.msgpack_restore((directory / 'codec_repaired.msgpack').read_bytes())
        assert set(repaired) == {'params', 'codebooks'}
        assert digest((directory / 'codec_repaired.msgpack').read_bytes()) == row['repaired_codec_sha256']
        for name in base['params']:
            if name != 'decoder':
                for a, b in zip(jax.tree.leaves(base['params'][name]), jax.tree.leaves(repaired['params'][name])):
                    np.testing.assert_array_equal(a, b)
    selected = output / 'seed-3' / 'dependency'
    (selected / 'decoder_repair.msgpack').write_bytes(snapshots[0])
    (selected / 'result.json').unlink()
    resumed = run(cfg, output, resume=True)
    assert resumed['followup_decision'] == report['followup_decision']
    for before, after in zip(report['results'], resumed['results']):
        for key in ('total_variation', 'zero_support_teacher_mass', 'repaired_codec_sha256',
                    'prior_sha256', 'generated_probability_with_other_bin'):
            assert before[key] == after[key], key


def test_checkpoint_hash_is_independent_of_mapping_insertion_order():
    a = {'z': jnp.array([1., 2.]), 'a': {'second': jnp.array(2), 'first': jnp.array(1)}}
    b = {'a': {'first': jnp.array(1), 'second': jnp.array(2)}, 'z': jnp.array([1., 2.])}
    assert state_bytes(a) == state_bytes(b)


def test_balanced_codec_sampler_preserves_empirical_dependence_weights():
    from ncm.coverage import codec_training_distribution
    observed = np.array([0, 0, 0, 0, 2, 2, 3])
    candidates, weights = codec_training_distribution(observed, 5, 'balanced_observed')
    np.testing.assert_array_equal(candidates, [0, 2, 3])
    np.testing.assert_allclose(weights[candidates] / len(candidates), [4/7, 2/7, 1/7], atol=1e-7)
    assert weights[1] == weights[4] == 0
    empirical, uniform_weights = codec_training_distribution(observed, 5)
    np.testing.assert_array_equal(empirical, observed)
    np.testing.assert_array_equal(uniform_weights, 1)


def test_encoder_collision_bound_exposes_irrecoverable_information_loss():
    from ncm.coverage import encoding_diagnostics
    stats = encoding_diagnostics(np.array([[0, 0], [0, 0], [1, 0]]), np.array([.4, .3, .3]))
    assert stats['population_unique_encodings'] == 2
    np.testing.assert_allclose(stats['max_sequence_reconstruction_from_codes'], .7)
    assert stats['code_collision_groups'] == [[0, 1]]
