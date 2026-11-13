"""Document-separated byte windows and reconstruction diagnostics.

JSONL records contain unique ``id`` and nonempty ``text`` fields. No window
crosses a document boundary. Incomplete tails are dropped, without UTF-8 repair.
"""
import hashlib
import json
from pathlib import Path
import string

import numpy as np


def read_documents(path):
    documents = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [item['id'] for item in documents]
    if not documents or len(set(ids)) != len(ids):
        raise ValueError('Documents must have unique IDs and be nonempty')
    if any(not isinstance(item['text'], str) or not item['text'] for item in documents):
        raise ValueError('Document text must be a nonempty string')
    return documents


def validate_splits(paths):
    """Reject identity or exact-content leakage across all supplied splits."""
    seen_ids, seen_content = set(), set()
    for path in paths:
        documents = read_documents(path)
        ids = {item['id'] for item in documents}
        content = {hashlib.sha256(item['text'].encode()).hexdigest() for item in documents}
        if seen_ids & ids or seen_content & content:
            raise ValueError('Document identity or content overlaps between splits')
        seen_ids.update(ids)
        seen_content.update(content)


def document_dataset(path, length):
    if length < 1:
        raise ValueError('Window length must be positive')
    rows, document_ids = [], []
    dropped = 0
    for item in read_documents(path):
        raw = item['text'].encode('utf-8')
        count = len(raw) // length
        dropped += len(raw) % length
        for offset in range(count):
            rows.append(np.frombuffer(raw[offset * length:(offset + 1) * length], np.uint8))
            document_ids.append(item['id'])
    if not rows:
        raise ValueError('No full windows at this length')
    tokens = np.asarray(rows, np.int32)
    data = tokens, np.ones_like(tokens, bool), hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return data, {'document_ids': document_ids, 'dropped_tail_bytes': dropped}


def reconstruction_metrics(target, prediction, mask, document_ids=None):
    target, prediction, mask = np.asarray(target), np.asarray(prediction), np.asarray(mask, bool)
    if target.shape != prediction.shape or target.shape != mask.shape or target.ndim != 2:
        raise ValueError('Expected matching [windows,length] arrays')
    if not mask.any() or np.any(~mask.any(axis=1)):
        raise ValueError('Every window must contain a valid byte')
    hits = target == prediction
    exact = np.all(hits | ~mask, axis=1)
    result = {'byte_accuracy': float(hits[mask].mean()),
              'exact_window_accuracy': float(exact.mean()),
              'windows': len(target), 'valid_bytes': int(mask.sum()),
              'errors_by_position': ((~hits) & mask).sum(axis=0).tolist(),
              'valid_bytes_by_position': mask.sum(axis=0).tolist()}
    slices = {'digits': list(range(48, 58)),
              'punctuation': list(string.punctuation.encode('ascii'))}
    result['slices'] = {}
    for name, values in slices.items():
        selected = np.isin(target, values) & mask
        result['slices'][name] = {'bytes': int(selected.sum()),
                                  'accuracy': float(hits[selected].mean()) if selected.any() else None}
    # Per-document sufficient statistics allow clustered uncertainty estimates.
    if document_ids is not None:
        if len(document_ids) != len(target):
            raise ValueError('One document ID is required per window')
        result['documents'] = []
        ids = np.asarray(document_ids)
        for identifier in sorted(set(document_ids)):
            selected = ids == identifier
            result['documents'].append({'id': identifier,
                'bytes': int(mask[selected].sum()),
                'correct_bytes': int((hits[selected] & mask[selected]).sum()),
                'windows': int(selected.sum()), 'exact_windows': int(exact[selected].sum())})
    return result
