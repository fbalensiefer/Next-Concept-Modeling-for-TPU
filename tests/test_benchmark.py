import json

import jax
import numpy as np
import pytest

from ncm.benchmark import quality_summary, run
from ncm.experiment import run as run_codec
from ncm.prior_experiment import run as run_prior


def test_common_quality_statistics_ignore_padding():
    reference = (np.array([[65, 66, 0], [65, 66, 0]]),
                 np.array([[1, 1, 0], [1, 1, 0]], bool))
    stats = quality_summary(np.array([[65, 66, -1], [65, 66, -1]]), reference)
    assert stats['byte_bigram_tv_to_heldout'] == 0
    assert stats['distinct_fraction'] == .5
    assert stats['valid_utf8_fraction'] == 1
    assert stats['sample_text'] == ['AB', 'AB']


def test_benchmark_backend_guard():
    other = 'tpu' if jax.default_backend() != 'tpu' else 'cpu'
    with pytest.raises(ValueError, match='no benchmark was run'):
        run({}, '/unused', require_backend=other)


def test_tiny_benchmark_report_includes_quality_and_synchronized_timings(tmp_path):
    train, valid = tmp_path / 'train.txt', tmp_path / 'valid.txt'
    train.write_text('abababab')
    valid.write_text('ababac')
    codec = tmp_path / 'codec'
    prior = tmp_path / 'prior'
    run_codec(dict(train_file=str(train), validation_file=str(valid), lengths=[1, 2, 4],
                   codebook_size=4, dim=4, width=8, heads=2, layers=1, batch_size=2,
                   seed=1, learning_rate=.001, steps=1, checkpoint_every=1), codec)
    run_prior(dict(codec_run=str(codec), batch_size=2, seed=1, learning_rate=.001,
                   steps=1, checkpoint_every=1, sample_count=2), prior)
    config = dict(prior_run=str(prior), ar_steps=1, ar_batch_size=2, ar_learning_rate=.001,
                  seed=1, batch_sizes=[1], warmup=1, repeats=2, quality_samples=2, temperature=1.)
    report = run(config, tmp_path / 'benchmark')
    assert report['greedy_reference_equivalent']
    assert len(report['quality']) == 3
    assert report['per_case_measured_peak_device_bytes'] is None
    assert len(report['implementation_sha256']) > 0
    for result in report['trained_results']:
        assert result['compile_seconds'] >= 0
        assert result['median_ms'] > 0
        assert len(result['latency_seconds']) == 2
        np.testing.assert_allclose(result['bytes_per_second'], 4 / (result['median_ms'] / 1000))
    saved = json.loads((tmp_path / 'benchmark' / 'report.json').read_text())
    assert saved['config']['prior_sha256'] == report['config']['prior_sha256']
    assert (tmp_path / 'benchmark' / 'ar_checkpoint.msgpack').exists()
    repeated = run(config, tmp_path / 'benchmark')
    assert repeated['ar_training']['resumed_from'] == 1
    assert repeated['quality'] == report['quality']
