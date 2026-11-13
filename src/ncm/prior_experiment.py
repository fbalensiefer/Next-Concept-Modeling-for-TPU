"""Train/evaluate a frozen-codec prior and save reproducible byte samples."""
import argparse
import hashlib
import json
from pathlib import Path

from .config import experiment_rvq_config

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import LossConfig, NCM, RVQConfig, TransformerConfig, initialize
from .experiment import batch_at, dataset
from .generation import PriorState, bytes_to_text, evaluate_prior, generate, make_prior_step
from .text_data import document_dataset, validate_splits


def load_codec(directory):
    directory = Path(directory)
    cfg = json.loads((directory / 'config.json').read_text())
    model = NCM(experiment_rvq_config(cfg),
                TransformerConfig(cfg['width'], cfg['heads'], cfg['layers']), 256,
                LossConfig(prior=0., entropy=cfg.get('entropy', .01)))
    batch = batch_at((np.zeros((1, cfg['lengths'][-1]), np.int32),
                      np.ones((1, cfg['lengths'][-1]), bool)), np.array([0]))
    template = initialize(model, optax.adam(cfg['learning_rate']), jax.random.key(cfg['seed']), batch, initialize_codebooks=False)
    raw = (directory / 'checkpoint.msgpack').read_bytes()
    return model, serialization.from_bytes(template, raw), cfg, hashlib.sha256(raw).hexdigest()


def run(config, output, resume=False):
    cfg = dict(config)
    model, codec, source_cfg, source_hash = load_codec(cfg['codec_run'])
    train_path = cfg.get('train_file', source_cfg['train_file'])
    valid_path = cfg.get('validation_file', source_cfg['validation_file'])
    data_format = cfg.get('data_format', source_cfg.get('data_format', 'text'))
    if data_format == 'documents_jsonl':
        paths = [train_path, valid_path]
        test_path = cfg.get('test_file', source_cfg.get('test_file'))
        if test_path:
            paths.append(test_path)
        validate_splits(paths)
        train, _ = document_dataset(train_path, model.rvq.max_length)
        valid, _ = document_dataset(valid_path, model.rvq.max_length)
    elif data_format == 'text':
        train = dataset(train_path, model.rvq.max_length)
        valid = dataset(valid_path, model.rvq.max_length)
    else:
        raise ValueError('Unknown data_format')
    if train[2] == valid[2]:
        raise ValueError('Training and validation files must differ')
    identity = dict(cfg, codec_sha256=source_hash, train_sha256=train[2], validation_sha256=valid[2])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'config.json'
    checkpoint = output / 'checkpoint.msgpack'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Prior configuration, source checkpoint or corpus changed')
    if checkpoint.exists() and not resume:
        raise ValueError('Use --resume or a new output directory')
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    optimizer = optax.adam(cfg['learning_rate'])
    params = codec.params['SharedLevelGenerator_0']
    state = PriorState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))
    if resume:
        state = serialization.from_bytes(state, checkpoint.read_bytes())
    codec_params = codec.params['MultiscaleTokenizer_0']
    step = jax.jit(make_prior_step(model, optimizer, codec_params, codec.codebooks))
    evaluate = jax.jit(lambda params, batch: evaluate_prior(
        model, codec_params, params, codec.codebooks, batch))

    def report(data):
        totals = None
        for start in range(0, len(data[0]), cfg['batch_size']):
            batch = batch_at(data, np.arange(start, min(start + cfg['batch_size'], len(data[0]))))
            stats = jax.tree.map(np.asarray, evaluate(state.params, batch))
            totals = stats if totals is None else jax.tree.map(lambda a, b: a + b, totals, stats)
        result = {}
        for prefix in ['teacher', 'free']:
            count = np.maximum(totals['count'], 1)
            result[prefix + '_code_ce'] = (totals[prefix + '_nll_sum'] / count).tolist()
            result[prefix + '_code_accuracy'] = (totals[prefix + '_hits'] / count).tolist()
            byte_name = 'codec_reconstruction_accuracy' if prefix == 'teacher' else 'free_reference_byte_accuracy'
            result[byte_name] = float(totals[prefix + '_byte_hits'] / totals['byte_count'])
        result['code_count'] = totals['count'].tolist()
        result['ce_degradation'] = (np.array(result['free_code_ce']) - result['teacher_code_ce']).tolist()
        return result

    seed = jax.random.key(cfg['seed'])
    for index in range(int(state.step), cfg['steps']):
        order_key, dropout_key = jax.random.split(jax.random.fold_in(seed, index + 1))
        indices = np.asarray(jax.random.randint(order_key, (cfg['batch_size'],), 0, len(train[0])))
        state, _ = step(state, batch_at(train, indices), dropout_key)
        if (index + 1) % cfg.get('checkpoint_every', 100) == 0 or index + 1 == cfg['steps']:
            temporary = output / 'checkpoint.tmp'
            temporary.write_bytes(serialization.to_bytes(state))
            temporary.replace(checkpoint)
            record = {'step': int(state.step), 'train': report(train), 'validation': report(valid)}
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    tokens, codes = generate(model, codec_params, state.params, codec.codebooks,
                             jax.random.key(cfg.get('sample_seed', 42)),
                             batch_size=cfg.get('sample_count', 16),
                             temperature=cfg.get('temperature', 1.), top_k=cfg.get('top_k', 0))
    samples = {'seed': cfg.get('sample_seed', 42), 'temperature': cfg.get('temperature', 1.),
               'top_k': cfg.get('top_k', 0), 'step': int(state.step),
               'text': bytes_to_text(tokens), 'bytes': np.asarray(tokens).tolist(),
               'codes': np.asarray(codes).tolist()}
    (output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output, args.resume)


if __name__ == '__main__':
    main()
