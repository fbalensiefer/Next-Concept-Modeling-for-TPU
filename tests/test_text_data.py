import json
import numpy as np
import pytest
from ncm.text_data import document_dataset, validate_splits, reconstruction_metrics


def write(path, docs):
    path.write_text(''.join(json.dumps(dict(id=i, text=t)) + '\n' for i, t in docs))
    return path


def test_windows_never_cross_documents(tmp_path):
    path = write(tmp_path / 'data.jsonl', [('a', 'abcde'), ('b', 'VWXYZ')])
    data, metadata = document_dataset(path, 4)
    assert [bytes(row).decode() for row in data[0].astype(np.uint8)] == ['abcd', 'VWXY']
    assert metadata == dict(document_ids=['a', 'b'], dropped_tail_bytes=2)
    assert data[1].all()


@pytest.mark.parametrize('docs', [[('a', 'different')], [('b', 'same')]])
def test_split_leakage_rejected(tmp_path, docs):
    first = write(tmp_path / 'train.jsonl', [('a', 'same')])
    second = write(tmp_path / 'valid.jsonl', docs)
    with pytest.raises(ValueError, match='overlaps'):
        validate_splits([first, second])


def test_metrics_preserve_failures_and_document_counts():
    target = np.array([[49, 33, 65], [50, 66, 0]])
    prediction = np.array([[49, 46, 65], [50, 66, 99]])
    mask = np.array([[True, True, True], [True, True, False]])
    result = reconstruction_metrics(target, prediction, mask, ['one', 'two'])
    assert result['byte_accuracy'] == .8
    assert result['exact_window_accuracy'] == .5
    assert result['slices']['digits']['accuracy'] == 1
    assert result['slices']['punctuation']['accuracy'] == 0
    assert result['documents'][0]['exact_windows'] == 0
    assert result['documents'][1]['correct_bytes'] == 2
