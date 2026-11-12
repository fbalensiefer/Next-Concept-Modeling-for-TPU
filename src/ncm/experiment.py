"""Reproducible UTF-8 byte reconstruction experiment (no fallback side channel)."""
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

from . import Batch, LossConfig, NCM, RVQConfig, TransformerConfig, initialize, make_train_step


def dataset(path, length):
    raw = Path(path).read_bytes()
    if not raw:
        raise ValueError('Text files must be nonempty')
    values = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    rows = (len(values) + length - 1) // length
    tokens = np.zeros(rows * length, np.int32)
    mask = np.arange(rows * length) < len(values)
    tokens[:len(values)] = values
    return tokens.reshape(rows, length), mask.reshape(rows, length), hashlib.sha256(raw).hexdigest()


def batch_at(data, indices):
    tokens, mask = (jnp.asarray(x[indices]) for x in data[:2])
    return Batch(tokens, mask, jnp.zeros_like(tokens), jnp.zeros_like(mask))


def run(config, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    train = dataset(cfg['train_file'], cfg['lengths'][-1])
    valid = dataset(cfg['validation_file'], cfg['lengths'][-1])
    if train[2] == valid[2]:
        raise ValueError('Training and validation files must differ')
    identity = dict(cfg, train_sha256=train[2], validation_sha256=valid[2])
    manifest = output / 'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Run configuration or dataset changed')
    if not resume and (output / 'checkpoint.msgpack').exists():
        raise ValueError('Existing checkpoint: use --resume or a new output directory')
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    model = NCM(experiment_rvq_config(cfg),
                TransformerConfig(cfg['width'], cfg['heads'], cfg['layers']), 256,
                LossConfig(prior=0., entropy=cfg.get('entropy', .01)))
    optimizer = optax.adam(cfg['learning_rate'])
    seed = jax.random.key(cfg['seed'])
    initial_batch = batch_at(train, np.arange(cfg['batch_size']) % len(train[0]))
    warmups = []
    if model.rvq.coarse.enabled and not resume:
        for index in range(model.rvq.coarse.warmup_batches):
            order_key, _ = jax.random.split(jax.random.fold_in(seed, index + 1))
            ids = np.asarray(jax.random.randint(order_key, (cfg['batch_size'],), 0, len(train[0])))
            warmups.append(batch_at(train, ids))
    state = initialize(model, optimizer, seed, initial_batch, warmup_batches=warmups or None,
                       initialize_codebooks=not resume)
    checkpoint = output / 'checkpoint.msgpack'
    if resume:
        state = serialization.from_bytes(state, checkpoint.read_bytes())
    step = jax.jit(make_train_step(model, optimizer))

    @jax.jit
    def evaluate(state, batch):
        variables = {'params': state.params['MultiscaleTokenizer_0']}
        from .tokenizer import MultiscaleTokenizer
        result = MultiscaleTokenizer(model.rvq, 256).apply(
            variables, batch.tokens, batch.mask, batch.fallback_bytes,
            batch.fallback_mask, state.codebooks)
        from .losses import cross_entropy
        hits = ((result.logits.argmax(-1) == batch.tokens) & batch.mask).sum()
        return cross_entropy(result.logits, batch.tokens, batch.mask), hits, batch.mask.sum()

    def report(data):
        loss = hits = count = 0.
        for i in range(0, len(data[0]), cfg['batch_size']):
            b = batch_at(data, np.arange(i, min(i + cfg['batch_size'], len(data[0]))))
            ce, h, n = map(float, evaluate(state, b))
            loss += ce * n
            hits += h
            count += n
        return {'byte_cross_entropy': loss / count, 'byte_accuracy': hits / count}

    for index in range(int(state.step), cfg['steps']):
        # Step-addressed randomness also reproduces batch order after resume.
        key = jax.random.fold_in(seed, index + 1)
        order_key, dropout_key = jax.random.split(key)
        indices = np.asarray(jax.random.randint(order_key, (cfg['batch_size'],), 0, len(train[0])))
        state, metrics = step(state, batch_at(train, indices), jnp.asarray(0), dropout_key)
        if (index + 1) % cfg.get('checkpoint_every', 100) == 0 or index + 1 == cfg['steps']:
            temporary = output / 'checkpoint.tmp'
            temporary.write_bytes(serialization.to_bytes(state))
            temporary.replace(checkpoint)
            record = {'step': int(state.step), 'train': report(train), 'validation': report(valid)}
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
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
