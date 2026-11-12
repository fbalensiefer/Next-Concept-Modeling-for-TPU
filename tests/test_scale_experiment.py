import json
from pathlib import Path

import numpy as np
import pytest

from ncm.scale_experiment import run
from flax import serialization
import ncm.scale_experiment as experiment_module


@pytest.mark.parametrize('stabilized', [False, True])
def test_scale_protocol_exact_distribution_and_mid_codec_resume(tmp_path, monkeypatch, stabilized):
    cfg = json.loads(Path('examples/scale/config.json').read_text())
    cfg.update(lengths=[1, 2], dim=4, width=8, teacher_width=8, seeds=[0],
               teacher_steps=2, codec_steps=2, prior_steps=2, batch_size=4,
               checkpoint_every=1, train_samples=16, validation_samples=16, calibration_samples=16)
    if stabilized:
        cfg.update(arms=['reconstruction', 'alignment', 'dependency', 'alignment_dependency'],
                   coarse=dict(enabled=True, warmup_batches=2, kmeans_iterations=3, usage_decay=0.,
                               dead_code_threshold=8., dead_code_grace_steps=0, temp_anneal_steps=1, beta_warmup_steps=1))
    snapshots = []
    initialization_hashes = {}
    warmup_calls = []
    warmup = experiment_module.warmup_codebooks
    def track_warmup(*args, **kwargs):
        warmup_calls.append(True)
        return warmup(*args, **kwargs)
    monkeypatch.setattr(experiment_module, 'warmup_codebooks', track_warmup)
    original = Path.replace
    def capture(path, target):
        result = original(path, target)
        if Path(target).name == 'codec.msgpack':
            raw = serialization.msgpack_restore(Path(target).read_bytes())
            if int(raw['step']) == 0:
                assert raw['opt_state']['coarse']['initialized']
                initialization_hashes[Path(target).parent.name] = Path(target).read_bytes()
            if Path(target).parent.name == 'reconstruction' and int(raw['step']) == 1 and not snapshots:
                snapshots.append(Path(target).read_bytes())
        return result
    monkeypatch.setattr(Path, 'replace', capture)
    output = tmp_path / 'pilot'
    first = run(cfg, output)
    assert len(first['results']) == (4 if stabilized else 3)
    if stabilized:
        assert len(warmup_calls) == 1
        assert len(set(initialization_hashes.values())) == 1
        assert len(initialization_hashes) == 4
        for row in first['results']:
            assert row['coarse_maintenance']['initialized']
            assert row['coarse_maintenance']['initial_samples'] == 8
            assert row['coarse_maintenance']['step'] == 2
            assert sum(row['coarse_maintenance']['replacements']) > 0
    assert first['decision']['matched_student_flop_estimates']
    for row in first['results']:
        np.testing.assert_allclose(sum(row['generated_probability_with_other_bin']), 1, atol=1e-6)
        np.testing.assert_allclose(row['hard_conditional_tc_sum'], row['independent_factorization_kl_nats'], atol=3e-6)
    arm = output / 'seed-0' / 'reconstruction'
    (arm / 'codec.msgpack').write_bytes(snapshots[0])
    (arm / 'prior.msgpack').unlink()
    (arm / 'result.json').unlink()
    resumed = run(cfg, output, resume=True)
    assert first['decision'] == resumed['decision']
    if stabilized:
        assert len(warmup_calls) == 1  # reload uses saved centroids/flag, not another warmup
    for before, after in zip(first['results'], resumed['results']):
        for key in before:
            if not key.endswith('_seconds_this_invocation'):
                assert before[key] == after[key], key
    with pytest.raises(ValueError, match='changed'):
        run(dict(cfg, dependency_weight=.5), output, resume=True)
    with pytest.raises(ValueError, match='Existing'):
        run(cfg, output)
