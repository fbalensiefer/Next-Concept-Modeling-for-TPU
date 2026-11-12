"""Sample fixed-length byte sequences from a trained frozen-codec prior."""
import argparse
import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import numpy as np

from .generation import bytes_to_text, generate
from .prior_experiment import load_codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--length', type=int)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-k', type=int, default=0)
    args = parser.parse_args()
    directory = Path(args.run)
    cfg = json.loads((directory / 'config.json').read_text())
    model, codec, _, source_hash = load_codec(cfg['codec_run'])
    if source_hash != cfg['codec_sha256']:
        raise ValueError('Source codec checkpoint changed')
    raw = (directory / 'checkpoint.msgpack').read_bytes()
    prior = serialization.msgpack_restore(raw)
    tokens, codes = generate(model, codec.params['MultiscaleTokenizer_0'], prior['params'],
                             codec.codebooks, jax.random.key(args.seed), batch_size=args.count,
                             length=args.length, temperature=args.temperature, top_k=args.top_k)
    print(json.dumps({'seed': args.seed, 'temperature': args.temperature, 'top_k': args.top_k,
                      'length': args.length or model.rvq.max_length,
                      'checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
                      'text': bytes_to_text(tokens), 'bytes': np.asarray(tokens).tolist(),
                      'codes': np.asarray(codes).tolist()}, indent=2))


if __name__ == '__main__':
    main()
