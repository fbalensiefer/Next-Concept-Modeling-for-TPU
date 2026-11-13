"""Bounded CPU reconstruction pilot, with book-separated Gutenberg text.

Download ebooks 11, 12 and 13 as UTF-8 into OUTPUT/raw/ID.txt first.
This runner never evaluates test reconstruction and refuses an existing protocol.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import jax
from ncm.experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    if args.steps < 1:
        raise ValueError('steps must be positive')
    if (output / 'protocol.json').exists():
        raise ValueError('Existing protocol; use a new output directory')
    corpus = []
    for split, identifier in [('train', 11), ('validation', 12), ('test', 13)]:
        raw = (output / 'raw' / f'{identifier}.txt').read_bytes()
        text = raw.decode('utf-8-sig').replace('\r\n', '\n')
        start = re.search(r'^\*\*\* START OF .*?\*\*\*.*\n', text, re.MULTILINE)
        end = re.search(r'^\*\*\* END OF .*?\*\*\*', text, re.MULTILINE)
        if not start or not end or start.end() >= end.start():
            raise ValueError('Unrecognized Gutenberg boundaries')
        body = text[start.end():end.start()].strip()
        processed = json.dumps(dict(id=f'gutenberg-{identifier}', text=body), ensure_ascii=False) + '\n'
        (output / f'{split}.jsonl').write_text(processed)
        corpus.append(dict(split=split, id=identifier,
            source=f'https://www.gutenberg.org/ebooks/{identifier}',
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            processed_sha256=hashlib.sha256(processed.encode()).hexdigest(),
            bytes=len(body.encode())))
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path('src/ncm').glob('*.py'))}
    sources[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = dict(corpus=corpus, lengths=[32, 128, 512], seeds=[0, 1, 2],
        steps=args.steps, batch_size=4, sources=sources,
        python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy')},
        devices=[str(device) for device in jax.devices()],
        selection='Final checkpoint; validation is descriptive; no test reconstruction',
        limitations='One book per split, one author, short bounded fits; not a quality or speed comparison')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    summary = []
    for length in protocol['lengths']:
        for seed in protocol['seeds']:
            config = dict(data_format='documents_jsonl',
                train_file=str(output / 'train.jsonl'), validation_file=str(output / 'validation.jsonl'),
                test_file=str(output / 'test.jsonl'), lengths=[length // 4, length // 2, length],
                codebook_size=16, dim=16, width=16, heads=2, layers=1,
                batch_size=4, seed=seed, learning_rate=.003, steps=args.steps,
                checkpoint_every=args.steps, entropy=0.)
            destination = output / f'length-{length}-seed-{seed}'
            started = time.perf_counter()
            state = run(config, destination)
            jax.block_until_ready(state)
            seconds = time.perf_counter() - started
            record = json.loads((destination / 'metrics.jsonl').read_text().splitlines()[-1])
            summary.append(dict(length=length, seed=seed, elapsed_seconds=seconds,
                training_byte_exposure=args.steps * 4 * length, metrics=record))
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
