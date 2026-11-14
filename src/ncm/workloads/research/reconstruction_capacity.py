"""Fixed-length training-duration and code-capacity follow-up.

Uses the processed book-separated pilot files without reading test text.
All six fits are fresh; changing codebook size is the sole architecture change.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import jax
import numpy as np
from ncm.experiment import run
from ncm.text_data import document_dataset, reconstruction_metrics, validate_splits
from ncm.representations import encode_pairs, decode_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='runs/real-text-pilot')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    if output.exists():
        raise ValueError('Use a fresh output directory')
    paths = [data / f'{split}.jsonl' for split in ('train', 'validation')]
    validate_splits(paths)
    if not (data / 'protocol.json').exists():
        raise ValueError('Missing source pilot protocol')
    output.mkdir(parents=True)
    sources = [*sorted(Path('src/ncm').glob('*.py')), Path(__file__).resolve().relative_to(Path.cwd())]
    for path in sources:
        target = output / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    configs = []
    for size in (16, 64):
        for seed in (0, 1, 2):
            configs.append(dict(data_format='documents_jsonl', train_file=str(paths[0]),
                validation_file=str(paths[1]), lengths=[32, 64, 128], codebook_size=size,
                dim=16, width=16, heads=2, layers=1, batch_size=4, seed=seed,
                learning_rate=.003, steps=2000, checkpoint_every=100, entropy=0.))
    protocol = dict(configs=configs, source_pilot=json.loads((data / 'protocol.json').read_text()),
        data_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        python=platform.python_version(), devices=[str(d) for d in jax.devices()],
        versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','flax','optax','numpy')},
        decision='Compare paired final fidelity and training curves; no threshold tuning or test evaluation',
        controls='Raw bytes and bijective byte pairs; nominal 8 bits per input byte in both',
        limitations='One validation book; exploratory follow-up; no priors, generation or TPU claims')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    controls = {}
    for split, path in zip(('train','validation'), paths):
        values, metadata = document_dataset(path, 128)
        tokens, mask = values[:2]
        controls[split] = {name: reconstruction_metrics(tokens, predicted, mask, metadata['document_ids'])
            for name, predicted in [('raw_bytes', tokens), ('fixed_pairs', decode_pairs(encode_pairs(tokens)))]}
    (output / 'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    summary = []
    for cfg in configs:
        destination = output / f"codes-{cfg['codebook_size']}-seed-{cfg['seed']}"
        started = time.perf_counter()
        state = run(cfg, destination)
        jax.block_until_ready(state)
        params = state.params['MultiscaleTokenizer_0']
        summary.append(dict(codebook_size=cfg['codebook_size'], seed=cfg['seed'],
            elapsed_seconds=time.perf_counter()-started,
            codec_parameters=sum(int(x.size) for x in jax.tree.leaves(params)),
            codebook_embedding_elements=int(state.codebooks.embeddings.size),
            nominal_code_bits_per_byte=sum(cfg['lengths'])*np.log2(cfg['codebook_size'])/128,
            records=[json.loads(line) for line in (destination/'metrics.jsonl').read_text().splitlines()]))
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
