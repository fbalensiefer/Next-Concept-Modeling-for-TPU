"""Deterministic local real-text windows, preserving document tails and hashes."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TextDataset:
    tokens: np.ndarray
    mask: np.ndarray
    sha256: str
    document_ids: tuple
    byte_counts: np.ndarray


def load_text(path, length, *, expected_sha256=None, tokenizer=None):
    """Load UTF-8 text or ID/text JSONL. Optional tokenizer exposes encode/decode.

    Subword byte counts are measured from decoded windows; byte mode counts
    original bytes exactly, including UTF-8 sequences split across windows.
    Hash verification happens before parsing or tokenization.
    """
    if length < 1:
        raise ValueError('length must be positive')
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('corpus SHA256 mismatch')
    if path.suffix == '.jsonl':
        documents = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    else:
        documents = [{'id': path.name, 'text': raw.decode('utf-8')}]
    ids = [d['id'] for d in documents]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate document IDs')
    rows, masks, owners, counts = [], [], [], []
    for doc in documents:
        tokens = (list(doc['text'].encode('utf-8')) if tokenizer is None
                  else list(tokenizer.encode(doc['text'])))
        for start in range(0, len(tokens), length):
            chunk = tokens[start:start + length]
            rows.append(chunk + [0] * (length - len(chunk)))
            masks.append([True] * len(chunk) + [False] * (length - len(chunk)))
            owners.append(doc['id'])
            counts.append(len(chunk) if tokenizer is None else len(tokenizer.decode(chunk).encode('utf-8')))
    if not rows:
        raise ValueError('empty corpus')
    return TextDataset(np.asarray(rows, np.int32), np.asarray(masks, bool), digest,
                       tuple(owners), np.asarray(counts, np.int32))
