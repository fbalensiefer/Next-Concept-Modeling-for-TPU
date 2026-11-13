import json
from pathlib import Path
import jax
import numpy as np
import pytest

from ncm.experiment import run as run_codec
from ncm.prior_experiment import run


@pytest.mark.parametrize('stabilized', [False, True])
def test_prior_checkpoint_resume_and_source_integrity(tmp_path, monkeypatch, stabilized):
    train, valid = tmp_path / 'train.txt', tmp_path / 'valid.txt'
    train.write_text('a cat sat. a cat sat.')
    valid.write_text('a cat ran.')
    codec_dir = tmp_path / 'codec'
    codec_config = dict(train_file=str(train), validation_file=str(valid), lengths=[1, 2, 4],
                        codebook_size=4, dim=4, width=8, heads=2, layers=1, batch_size=2,
                        seed=1, learning_rate=.001, steps=2, checkpoint_every=2)
    if stabilized:
        codec_config['coarse'] = dict(enabled=True, warmup_batches=2, kmeans_iterations=3)
    run_codec(codec_config, codec_dir)
    source_bytes = (codec_dir / 'checkpoint.msgpack').read_bytes()
    saved = []
    original = Path.replace

    def capture(path, target):
        result = original(path, target)
        if not saved:
            saved.append(Path(target).read_bytes())
        return result

    monkeypatch.setattr(Path, 'replace', capture)
    config = dict(codec_run=str(codec_dir), batch_size=2, seed=3,
                  learning_rate=.001, steps=4, checkpoint_every=2, sample_count=4)
    output = tmp_path / 'prior'
    full = run(config, output)
    samples = (output / 'samples.json').read_bytes()
    (output / 'checkpoint.msgpack').write_bytes(saved[0])
    resumed = run(config, output, resume=True)
    for a, b in zip(jax.tree.leaves(full), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)
    assert (output / 'samples.json').read_bytes() == samples
    assert (codec_dir / 'checkpoint.msgpack').read_bytes() == source_bytes
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1] == metrics[1]
    import pytest
    with pytest.raises(ValueError, match='changed'):
        run(dict(config, seed=4), output, resume=True)


def test_document_prior_inherits_codec_format_and_rejects_leakage(tmp_path):
    paths = {}
    for split, text in [('train', 'abcd' * 4), ('validation', 'abce' * 4), ('test', 'sealed')]:
        path = tmp_path / f'{split}.jsonl'
        path.write_text(json.dumps(dict(id=split, text=text)) + '\n')
        paths[split] = str(path)
    codec_dir = tmp_path / 'codec'
    run_codec(dict(data_format='documents_jsonl', train_file=paths['train'],
        validation_file=paths['validation'], test_file=paths['test'], lengths=[1,2,4],
        codebook_size=4, dim=4, width=8, heads=2, layers=1, batch_size=2,
        seed=1, learning_rate=.001, steps=1, checkpoint_every=1, entropy=0.), codec_dir)
    original = (codec_dir / 'checkpoint.msgpack').read_bytes()
    cfg = dict(codec_run=str(codec_dir), batch_size=2, seed=3,
        learning_rate=.001, steps=1, checkpoint_every=1, sample_count=2)
    run(cfg, tmp_path / 'prior')
    record = json.loads((tmp_path / 'prior' / 'metrics.jsonl').read_text())
    # Four byte windows, not JSON serialization bytes: 4*[1,2,4] codes.
    assert record['train']['code_count'] == [4,8,16]
    assert (codec_dir / 'checkpoint.msgpack').read_bytes() == original
    with pytest.raises(ValueError, match='overlaps'):
        run(dict(cfg, validation_file=paths['train']), tmp_path / 'leaky')
