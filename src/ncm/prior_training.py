"""Train a categorical causal prior over a frozen multiscale RVQ codec."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct

from .autoregressive import AutoregressiveCategorical
from .config import LossConfig, PriorTrainingConfig, TransformerConfig, experiment_rvq_config
from .experiment import batch_at, dataset
from .losses import cross_entropy
from .training import NCM, initialize


@struct.dataclass
class PriorTrainState:
    params: object
    opt_state: object
    step: jax.Array


def flatten_rvq_codes(indices, mask, lengths=None):
    """Serialize [K,B,N] RVQ streams in level-major order; keep valid positions."""
    if indices.ndim != 3 or mask.shape != indices.shape:
        raise ValueError('RVQ indices and mask must have matching [K,B,N] shapes')
    if lengths is None:
        codes = jnp.transpose(indices, (1, 0, 2)).reshape(indices.shape[1], -1)
        valid = jnp.transpose(mask, (1, 0, 2)).reshape(mask.shape[1], -1)
    else:
        if len(lengths) != indices.shape[0] or any(n <= 0 or n > indices.shape[2] for n in lengths):
            raise ValueError('lengths must specify a valid prefix for every RVQ level')
        codes = jnp.concatenate([indices[level, :, :length] for level, length in enumerate(lengths)], axis=1)
        valid = jnp.concatenate([mask[level, :, :length] for level, length in enumerate(lengths)], axis=1)
    return jnp.where(valid, codes, 0).astype(jnp.int32), valid.astype(bool)


def shifted_categorical_inputs(targets, vocab_size):
    """Teacher-forcing inputs with a dedicated BOS symbol equal to vocab_size."""
    bos = jnp.full((targets.shape[0], 1), vocab_size, dtype=jnp.int32)
    return jnp.concatenate((bos, targets[:, :-1]), axis=1)


def make_categorical_prior_train_step(prior, optimizer, prior_loss_weight):
    """Update only the prior parameter tree; codec weights are not optimizer inputs."""
    training = PriorTrainingConfig(prior_loss_weight=prior_loss_weight, freeze_codec=True)

    def step(state, targets, valid, reconstruction_loss=0., commitment_loss=0.):
        inputs = shifted_categorical_inputs(targets, prior.vocab_size)

        def objective(params):
            logits = prior.apply({'params': params}, inputs, valid)
            loss = cross_entropy(logits, targets, valid)
            total = reconstruction_loss + commitment_loss + training.prior_loss_weight * loss
            return total, loss

        (_, prior_loss), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return PriorTrainState(params, opt_state, state.step + 1), {
            'prior_loss': prior_loss,
            'reconstruction_loss': jnp.asarray(reconstruction_loss),
            'commitment_loss': jnp.asarray(commitment_loss),
            'weighted_prior_loss': training.prior_loss_weight * prior_loss,
            'total_loss': reconstruction_loss + commitment_loss + training.prior_loss_weight * prior_loss,
            'grad_norm': optax.global_norm(grads),
        }

    return step


def load_phase1_codec(config, checkpoint_path, seed):
    """Initialize an exact Phase 1 state template and strictly restore its checkpoint."""
    from .engine_rpc import robust_load_checkpoint

    cfg = dict(config)
    model = NCM(experiment_rvq_config(cfg),
                TransformerConfig(cfg['width'], cfg['heads'], cfg['layers']), 256,
                LossConfig(prior=0., entropy=cfg.get('entropy', .01)))
    optimizer = optax.adam(cfg['learning_rate'])
    sample = jnp.zeros((1, model.rvq.max_length), jnp.int32)
    from .training import Batch
    batch = Batch(sample, jnp.ones_like(sample, bool), jnp.zeros_like(sample),
                  jnp.zeros_like(sample, bool))
    template = initialize(model, optimizer, jax.random.key(seed), batch,
                          initialize_codebooks=False)
    state = robust_load_checkpoint(template, checkpoint_path)
    return model, state


def run(config, output):
    """Run prior-only training from a verified reconstruction checkpoint.

    Input and output paths are explicit. The Phase 1 checkpoint is read only;
    the prior checkpoint and metrics are written beneath ``output``.
    """
    controls = PriorTrainingConfig(config.get('prior_loss_weight', 1.),
                                   config.get('freeze_codec', True))
    if not controls.freeze_codec:
        raise ValueError('this prior trainer currently requires freeze_codec=true')
    codec_path = Path(config['codec_checkpoint'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config.get('data_format', 'text') == 'documents_jsonl':
        from .text_data import document_dataset
        train, _ = document_dataset(config['train_file'], config['lengths'][-1])
    elif config.get('data_format', 'text') == 'text':
        train = dataset(config['train_file'], config['lengths'][-1])
    else:
        raise ValueError('data_format must be text or documents_jsonl')
    model, codec_state = load_phase1_codec(config, codec_path, config['seed'])

    from .tokenizer import MultiscaleTokenizer
    tokenizer = MultiscaleTokenizer(model.rvq, 256)
    encode = jax.jit(lambda batch: tokenizer.apply(
        {'params': codec_state.params['MultiscaleTokenizer_0']}, batch.tokens, batch.mask,
        batch.fallback_bytes, batch.fallback_mask, codec_state.codebooks))

    def encode_batch(batch):
        encoded = encode(batch)
        targets, valid = flatten_rvq_codes(encoded.rvq.indices, encoded.rvq.mask,
                                           model.rvq.lengths)
        reconstruction = cross_entropy(encoded.logits, batch.tokens, batch.mask)
        commitment = (encoded.rvq.commitment_numerator.sum()
                      / jnp.maximum(encoded.rvq.commitment_denominator.sum(), 1))
        return targets, valid, reconstruction, config.get('beta', .25) * commitment

    max_codes = sum(model.rvq.lengths)
    transformer = TransformerConfig(config['width'], config['heads'], config['layers'])
    prior = AutoregressiveCategorical(transformer, max_codes, model.rvq.codebook_size)
    init_tokens = jnp.zeros((config['batch_size'], max_codes), jnp.int32)
    init_mask = jnp.ones_like(init_tokens, bool)
    params = prior.init(jax.random.key(config['seed'] + 1),
                        shifted_categorical_inputs(init_tokens, prior.vocab_size), init_mask)['params']
    optimizer = optax.adam(config['learning_rate'])
    state = PriorTrainState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))
    if config.get('prior_checkpoint'):
        from .engine_rpc import robust_load_checkpoint
        state = robust_load_checkpoint(state, config['prior_checkpoint'])
    step = jax.jit(make_categorical_prior_train_step(
        prior, optimizer, controls.prior_loss_weight))

    records = []
    for index in range(config['steps']):
        key = jax.random.fold_in(jax.random.key(config['seed']), index)
        ids = np.asarray(jax.random.randint(key, (config['batch_size'],), 0, len(train[0])))
        targets, valid, reconstruction, commitment = encode_batch(batch_at(train, ids))
        state, metrics = step(state, targets, valid, reconstruction, commitment)
        record = {'step': int(state.step), **{k: float(v) for k, v in metrics.items()}}
        records.append(record)
        if (index + 1) % config.get('log_every', 10) == 0 or index + 1 == config['steps']:
            print(json.dumps(record), flush=True)

    metadata = {
        'codec_checkpoint': str(codec_path.resolve()),
        'codec_step': int(codec_state.step),
        'prior_loss_weight': controls.prior_loss_weight,
        'freeze_codec': controls.freeze_codec,
        'codebook_size': model.rvq.codebook_size,
        'levels': model.rvq.levels,
        'max_sequence_length': max_codes,
        'steps': config['steps'],
        'final': records[-1] if records else None,
    }
    temporary = output / 'prior_checkpoint.tmp'
    temporary.write_bytes(serialization.to_bytes(state))
    temporary.replace(output / 'prior_checkpoint.msgpack')
    (output / 'prior_metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return state, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)


if __name__ == '__main__':
    main()
