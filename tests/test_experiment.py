import json
from pathlib import Path
import jax
import numpy as np
import pytest
from ncm.experiment import run


@pytest.mark.parametrize('stabilized', [False, True])
def test_checkpoint_resume_matches_uninterrupted(tmp_path, monkeypatch, stabilized):
    train, valid = tmp_path / 'train.txt', tmp_path / 'valid.txt'
    train.write_text('a cat sat. a cat sat.')
    valid.write_text('a cat ran.')
    config = dict(train_file=str(train), validation_file=str(valid), lengths=[1, 2, 4],
                  codebook_size=4, dim=4, width=8, heads=2, layers=1, batch_size=2,
                  seed=1, learning_rate=.001, steps=4, checkpoint_every=2)
    if stabilized:
        config['coarse'] = dict(enabled=True, warmup_batches=2, kmeans_iterations=3,
                                temp_anneal_steps=2, beta_warmup_steps=2, dead_code_grace_steps=0,
                                dead_code_threshold=4., usage_decay=0.)
    saved = []
    original = Path.replace

    def capture(path, target):
        result = original(path, target)
        if not saved:
            saved.append(Path(target).read_bytes())
        return result

    monkeypatch.setattr(Path, 'replace', capture)
    output = tmp_path / 'run'
    full = run(config, output)
    (output / 'checkpoint.msgpack').write_bytes(saved[0])
    resumed = run(config, output, resume=True)
    for expected, actual in zip(jax.tree.leaves(full), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(expected, actual)
    if stabilized:
        assert full.opt_state['coarse'].initialized
        assert full.opt_state['coarse'].step == full.step == 4
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert records[-1] == records[1]
