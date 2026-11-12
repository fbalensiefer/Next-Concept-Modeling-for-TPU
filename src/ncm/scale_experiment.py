"""Small finite-domain SCALE pilot with exact dependence and output-law checks.

This tests one teacher and one fixed-size hierarchy, not cross-size LLM transfer.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import Batch, NCM, RVQConfig, TransformerConfig, TrainState, init_codebooks
from .autoregressive import AutoregressiveBytes, shifted_inputs
from .config import experiment_rvq_config
from .coarse import ensure_initialized, maintain_coarse, optimizer_state, schedules, warmup_codebooks
from .coverage import (FrozenCodec, codec_training_distribution, encoding_diagnostics,
                       make_decoder_repair_step)
from .dependence import (codewords, conditional_tc, direct_factorization_kl, hard_joint,
                         soft_assignments, soft_joint)
from .generation import PriorState, evaluate_prior, make_prior_step
from .generator import SharedLevelGenerator
from .losses import cross_entropy, entropy_loss
from .rvq import ema_update
from .tokenizer import MultiscaleTokenizer


ARMS = ('reconstruction', 'alignment', 'alignment_dependency')
ALL_ARMS = ('reconstruction', 'alignment', 'dependency', 'alignment_dependency')


def arm_weights(arm, config):
    if arm not in ALL_ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    return jnp.asarray([config['alignment_weight'] if arm in ('alignment', 'alignment_dependency') else 0.,
                        config['dependency_weight'] if arm in ('dependency', 'alignment_dependency') else 0.], jnp.float32)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()



def state_bytes(state):
    """Canonicalize mapping order so checkpoint hashes survive template reloads."""
    return serialization.to_bytes(jax.tree.map(lambda leaf: leaf, state))


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(state_bytes(state))
    temporary.replace(path)


def as_batch(tokens):
    tokens = jnp.asarray(tokens, jnp.int32)
    return Batch(tokens, jnp.ones_like(tokens, bool), jnp.zeros_like(tokens), jnp.zeros_like(tokens, bool))


def fit_canonical(features, dim):
    """Training-only PCA feature interface; no claim of operator equivalence."""
    flat = np.asarray(features, np.float64).reshape(-1, features.shape[-1])
    if not 1 <= dim <= flat.shape[-1]:
        raise ValueError('Canonical dimension must fit teacher width')
    mean = flat.mean(0)
    _, singular, right = np.linalg.svd(flat - mean, full_matrices=False)
    projection = right[:dim].T
    # Fix arbitrary SVD signs for reproducibility on the same runtime.
    signs = np.sign(projection[np.argmax(np.abs(projection), axis=0), np.arange(dim)])
    projection *= np.where(signs == 0, 1, signs)
    scale = np.maximum(singular[:dim] / max(len(flat) - 1, 1)**.5, .1)
    return mean.astype(np.float32), projection.astype(np.float32), scale.astype(np.float32)


def make_codec_step(config, optimizer):
    rvq = experiment_rvq_config(config)
    tokenizer = MultiscaleTokenizer(rvq, 256)
    levels = jnp.asarray(tuple(range(rvq.levels, 0, -1)), jnp.float32)

    def step(state, batch, teacher_features, weights, example_weights=None):
        original = state
        sample_weights = jnp.ones(batch.tokens.shape[0]) if example_weights is None else example_weights
        temperature = beta = None
        if rvq.coarse.enabled:
            books, coarse = ensure_initialized(tokenizer, state.params, state.codebooks,
                                               state.opt_state['coarse'], batch)
            state = state.replace(codebooks=books, opt_state=dict(state.opt_state, coarse=coarse))
            temperature, beta = schedules(rvq.coarse, coarse)
        def objective(params):
            output = tokenizer.apply({'params': params}, batch.tokens, batch.mask,
                                     batch.fallback_bytes, batch.fallback_mask, state.codebooks,
                                     coarse_temperature=temperature)
            recon = cross_entropy(output.logits, batch.tokens, batch.mask)
            commitment = output.rvq.commitment_numerator.sum() / jnp.maximum(output.rvq.commitment_denominator.sum(), 1)
            entropy = entropy_loss(output.rvq.soft_counts, levels)
            alignment = jnp.mean((output.latent - teacher_features)**2)
            probabilities = soft_assignments(output.latent, state.codebooks, rvq, config['soft_temperature'],
                                             coarse_temperature=temperature)
            joint = soft_joint(probabilities, sample_weights, rvq.lengths, rvq.codebook_size)
            dependence = conditional_tc(joint, rvq.lengths, rvq.codebook_size).sum()
            weighted_commitment = .25 * commitment
            coarse_entropy = jnp.asarray(0., jnp.float32)
            if rvq.coarse.enabled:
                coefficients = jnp.full((rvq.levels,), .25).at[0].set(beta)
                weighted_commitment = (coefficients * output.rvq.commitment_numerator).sum() / jnp.maximum(output.rvq.commitment_denominator.sum(), 1)
                coarse_entropy = entropy_loss(output.rvq.soft_counts[:1], jnp.ones(1))
            loss = (recon + weighted_commitment + config['entropy_weight'] * entropy
                    + rvq.coarse.entropy_weight * coarse_entropy
                    + weights[0] * alignment + weights[1] * dependence)
            return loss, (output.rvq, output.latent, dict(loss=loss, reconstruction=recon, alignment=alignment,
                                                        soft_conditional_tc=dependence))
        (_, (rvq_output, latent, metrics)), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        neural_state = state.opt_state['neural'] if rvq.coarse.enabled else state.opt_state
        updates, neural_state = optimizer.update(grads, neural_state, state.params)
        if rvq.coarse.enabled:
            books, coarse, diagnostics = maintain_coarse(
                state.codebooks, state.opt_state['coarse'], rvq_output.counts, rvq_output.sums,
                latent, batch.mask & ~batch.fallback_mask, rvq)
            opt_state = {'neural': neural_state, 'coarse': coarse}
            metrics = dict(metrics, **diagnostics, coarse_temperature=temperature, coarse_beta=beta)
        else:
            opt_state = neural_state
            books = ema_update(state.codebooks, rvq_output.counts, rvq_output.sums, rvq)
        candidate = TrainState(optax.apply_updates(state.params, updates), opt_state, books, state.step + 1)
        updated = jax.lax.cond(jnp.any(batch.mask), lambda: candidate, lambda: original)
        return updated, metrics
    return step


def teacher_setup(config, output, resume):
    length = config['lengths'][-1]
    words = codewords((length,), 2)
    tokens = words + 65
    noise = config['source_noise']
    source_p = np.full(len(words), noise / len(words), np.float64)
    source_p[0] += (1 - noise) / 2
    source_p[-1] += (1 - noise) / 2
    teacher = AutoregressiveBytes(TransformerConfig(config['teacher_width'], 2, 1), length)
    all_batch = as_batch(tokens)
    key = jax.random.key(config['teacher_seed'])
    params = teacher.init(key, shifted_inputs(all_batch.tokens), all_batch.mask)['params']
    optimizer = optax.adam(config['teacher_learning_rate'])
    state = PriorState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))
    path = output / 'teacher.msgpack'
    if resume and path.exists():
        state = serialization.from_bytes(state, path.read_bytes())

    @jax.jit
    def step(state, batch):
        def objective(params):
            logits = teacher.apply({'params': params}, shifted_inputs(batch.tokens), batch.mask)[..., 65:67]
            return cross_entropy(logits, batch.tokens - 65, batch.mask)
        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        return PriorState(optax.apply_updates(state.params, updates), opt_state, state.step + 1), loss

    start = time.perf_counter()
    for index in range(int(state.step), config['teacher_steps']):
        rng = np.random.default_rng(np.random.SeedSequence([config['teacher_seed'], index, 10]))
        ids = rng.choice(len(words), config['batch_size'], p=source_p)
        state, _ = step(state, as_batch(tokens[ids]))
        if (index + 1) % config['checkpoint_every'] == 0 or index + 1 == config['teacher_steps']:
            save_state(path, state)
    jax.block_until_ready(state)
    teacher_seconds = time.perf_counter() - start
    logits = teacher.apply({'params': state.params}, shifted_inputs(all_batch.tokens), all_batch.mask)[..., 65:67]
    log_prob = np.asarray(jax.nn.log_softmax(logits), np.float64)
    p = np.exp(np.take_along_axis(log_prob, words[..., None], -1)[..., 0].sum(-1))
    if not np.isclose(p.sum(), 1., atol=3e-6):
        raise ValueError('Enumerated teacher probabilities do not normalize')
    p /= p.sum()
    hidden = np.asarray(teacher.apply({'params': state.params}, shifted_inputs(all_batch.tokens),
                                     all_batch.mask, method=teacher.represent))
    draws = {}
    for name, count, stream in [('train', config['train_samples'], 20),
                                ('validation', config['validation_samples'], 21),
                                ('calibration', config['calibration_samples'], 22)]:
        rng = np.random.default_rng(np.random.SeedSequence([config['teacher_seed'], stream]))
        draws[name] = rng.choice(len(words), count, p=p)
    mean, projection, scale = fit_canonical(hidden[draws['calibration']], config['dim'])
    canonical = ((hidden - mean) @ projection) / scale
    np.savez(output / 'teacher_domain.npz', tokens=tokens, probability=p, source_probability=source_p,
             canonical=canonical, mean=mean, projection=projection, scale=scale, **draws)
    return tokens, p, canonical, draws, {
        'checkpoint_sha256': digest(path.read_bytes()), 'teacher_steps': config['teacher_steps'],
        'training_seconds_this_invocation': teacher_seconds,
        'source_to_teacher_kl': float(np.sum(source_p[source_p > 0] * np.log(source_p[source_p > 0] / p[source_p > 0]))),
        'split_id_sha256': {name: digest(ids.tobytes()) for name, ids in draws.items()},
        'definition': 'Tiny locally trained AR teacher, normalized over A/B at every prefix.'}


def exact_output_distribution(model, codec, prior_params):
    """Enumerate every codeword, then push the prior law through byte argmax."""
    rvq = model.rvq
    words = codewords(rvq.lengths, rvq.codebook_size)
    size = rvq.max_length
    indices = np.full((rvq.levels, len(words), size), -1, np.int32)
    offset = 0
    for level, count in enumerate(rvq.lengths):
        indices[level, :, :count] = words[:, offset:offset + count]
        offset += count
    mask = jnp.asarray(indices >= 0)
    result = SharedLevelGenerator(rvq, model.transformer).apply(
        {'params': prior_params}, codec.codebooks, mask, jnp.asarray(indices),
        source_mask=jnp.ones((len(words), size), bool))
    log_prob = jax.nn.log_softmax(result.logits)
    chosen = jnp.take_along_axis(log_prob, jnp.maximum(jnp.asarray(indices), 0)[..., None], -1)[..., 0]
    code_probability = np.exp(np.asarray(jnp.where(mask, chosen, 0).sum(axis=(0, 2)), np.float64))
    if not np.isclose(code_probability.sum(), 1., atol=3e-6):
        raise ValueError('Enumerated prior probabilities do not normalize')
    code_probability /= code_probability.sum()
    logits = MultiscaleTokenizer(rvq, 256).apply(
        {'params': codec.params}, result.representation, jnp.ones((len(words), size), bool),
        method=MultiscaleTokenizer.decode)
    decoded = np.asarray(logits.argmax(-1))
    valid = np.all((decoded >= 65) & (decoded <= 66), axis=-1)
    ids = ((decoded - 65) * (2**np.arange(size - 1, -1, -1))).sum(-1)
    ids = np.where(valid, ids, 2**size)
    output = np.zeros(2**size + 1, np.float64)
    np.add.at(output, ids, code_probability)
    return output


def distribution_metrics(target, generated):
    target = np.append(target, 0.)
    midpoint = (target + generated) / 2
    def kl(p, q):
        selected = p > 0
        return float(np.sum(p[selected] * np.log(p[selected] / np.maximum(q[selected], 1e-300))))
    missing = float(target[generated == 0].sum())
    return {'total_variation': float(np.abs(target - generated).sum() / 2),
            'jensen_shannon_nats': .5 * (kl(target, midpoint) + kl(generated, midpoint)),
            'forward_kl_nats': None if missing > 0 else kl(target, generated),
            'zero_support_teacher_mass': missing, 'outside_domain_mass': float(generated[-1])}


def evaluate(config, model, codec, prior, tokens, p, canonical, validation_ids):
    batch = as_batch(tokens)
    output = MultiscaleTokenizer(model.rvq, 256).apply(
        {'params': codec.params}, batch.tokens, batch.mask, batch.fallback_bytes, batch.fallback_mask, codec.codebooks)
    hard_codes = np.concatenate([np.asarray(output.rvq.indices[k, :, :n])
                                 for k, n in enumerate(model.rvq.lengths)], axis=1)
    joint = hard_joint(jnp.asarray(hard_codes), jnp.asarray(p), model.rvq.lengths, model.rvq.codebook_size)
    tc = np.asarray(conditional_tc(joint, model.rvq.lengths, model.rvq.codebook_size))
    independent_kl = direct_factorization_kl(joint, model.rvq.lengths, model.rvq.codebook_size)
    if not np.isclose(tc.sum(), independent_kl, atol=3e-6):
        raise ValueError('Independent TC identity check failed')
    predicted = np.asarray(output.logits.argmax(-1))
    recon_by_sequence = np.mean(predicted == tokens, axis=-1)
    sequence_em = np.all(predicted == tokens, axis=-1)
    coarse_mass = np.zeros(model.rvq.codebook_size, np.float64)
    coarse_codes = np.asarray(output.rvq.indices[0, :, :model.rvq.lengths[0]])
    np.add.at(coarse_mass, coarse_codes.reshape(-1), np.repeat(p, model.rvq.lengths[0]) / model.rvq.lengths[0])
    coarse_entropy = -float(np.sum(coarse_mass * np.log(np.maximum(coarse_mass, 1e-30))))
    generated = exact_output_distribution(model, codec, prior.params)
    stats = evaluate_prior(model, codec.params, prior.params, codec.codebooks, as_batch(tokens[validation_ids]))
    count = np.maximum(np.asarray(stats['count']), 1)
    feature_error = np.asarray(jnp.mean((output.latent - jnp.asarray(canonical))**2, axis=(1, 2)))
    return {**encoding_diagnostics(hard_codes, p),
            'population_coarse_code_probability': coarse_mass.tolist(),
            'population_coarse_active_codes': int(np.count_nonzero(coarse_mass)),
            'population_coarse_perplexity': float(np.exp(coarse_entropy)),
            'population_byte_reconstruction': float(p @ recon_by_sequence),
            'population_sequence_reconstruction': float(p @ sequence_em),
            'validation_byte_reconstruction': float(recon_by_sequence[validation_ids].mean()),
            'population_alignment_mse': float(p @ feature_error),
            'hard_conditional_tc_nats': tc.tolist(), 'hard_conditional_tc_sum': float(tc.sum()),
            'independent_factorization_kl_nats': independent_kl,
            'teacher_code_ce': (np.asarray(stats['teacher_nll_sum']) / count).tolist(),
            'free_code_ce': (np.asarray(stats['free_nll_sum']) / count).tolist(),
            'hard_code_entropy_nats': float(-np.sum(np.asarray(joint) * np.log(np.maximum(np.asarray(joint), 1e-30)))),
            'teacher_probability': p.tolist(), 'generated_probability_with_other_bin': generated.tolist(),
            **distribution_metrics(p, generated)}



def repair_decoder(config, model, codec, tokens, train_ids, seed, directory, resume):
    """Uniformly replay distinct observed training sequences; freeze all other state."""
    tokenizer = MultiscaleTokenizer(model.rvq, 256)
    distinct = np.unique(train_ids)
    batch = as_batch(tokens[distinct])
    encoded = tokenizer.apply({'params': codec.params}, batch.tokens, batch.mask,
                              batch.fallback_bytes, batch.fallback_mask, codec.codebooks)
    representations = jax.lax.stop_gradient(encoded.representation)
    optimizer = optax.adam(config['decoder_repair_learning_rate'])
    params = codec.params['decoder']
    repair = PriorState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))
    path = directory / 'decoder_repair.msgpack'
    if resume and path.exists():
        repair = serialization.from_bytes(repair, path.read_bytes())
    step = jax.jit(make_decoder_repair_step(model.rvq.dim, optimizer))
    example_ids = np.arange(config['batch_size']) % len(distinct)
    executable = step.lower(repair, representations[example_ids], batch.tokens[example_ids]).compile()
    flops = executable.cost_analysis().get('flops')
    start = time.perf_counter()
    for index in range(int(repair.step), config['decoder_repair_steps']):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 32]))
        ids = rng.integers(0, len(distinct), size=config['batch_size'])
        repair, loss = executable(repair, representations[ids], batch.tokens[ids])
        if (index + 1) % config['checkpoint_every'] == 0 or index + 1 == config['decoder_repair_steps']:
            save_state(path, repair)
            print(json.dumps({'seed': seed, 'arm': directory.name, 'stage': 'decoder_repair',
                              'step': index + 1, 'loss': float(loss)}), flush=True)
    updated = FrozenCodec(dict(codec.params, decoder=repair.params), codec.codebooks)
    # Only decoder parameters can change; confirm the discrete encoder stays fixed.
    other_before = {k: v for k, v in codec.params.items() if k != 'decoder'}
    other_after = {k: v for k, v in updated.params.items() if k != 'decoder'}
    if digest(state_bytes(other_before)) != digest(state_bytes(other_after)):
        raise ValueError('Decoder repair changed encoder parameters')
    if digest(state_bytes(codec.codebooks)) != digest(state_bytes(updated.codebooks)):
        raise ValueError('Decoder repair changed codebook state')
    encoded_after = tokenizer.apply({'params': updated.params}, batch.tokens, batch.mask,
                                    batch.fallback_bytes, batch.fallback_mask, updated.codebooks)
    if not np.array_equal(encoded.rvq.indices, encoded_after.rvq.indices):
        raise ValueError('Decoder repair changed training code assignments')
    save_state(directory / 'codec_repaired.msgpack', updated)
    return updated, {'steps': int(repair.step), 'distinct_training_sequences': len(distinct),
                     'step_flops_estimate': flops, 'seconds_this_invocation': time.perf_counter() - start,
                     'encoder_and_codebooks_unchanged': True,
                     'selection': 'Uniform over distinct training observations; no validation/population weighting.'}


def followup_decision(rows, config):
    seeds = config['confirmation_seeds']
    paired = []
    for seed in seeds:
        arms = {r['arm']: r for r in rows if r['seed'] == seed}
        full, dep = arms['alignment_dependency'], arms['dependency']
        before = full['before_repair']
        paired.append({'seed': seed,
                       'alignment_tv_improvement_over_dependency_only': dep['total_variation'] - full['total_variation'],
                       'dependency_tv_improvement_without_alignment': arms['reconstruction']['total_variation'] - dep['total_variation'],
                       'repair_tv_change': full['total_variation'] - before['total_variation'],
                       'missing_teacher_mass_before': before['zero_support_teacher_mass'],
                       'missing_teacher_mass_after': full['zero_support_teacher_mass'],
                       'sequence_reconstruction_after': full['population_sequence_reconstruction']})
    alignment_mean = float(np.mean([r['alignment_tv_improvement_over_dependency_only'] for r in paired]))
    alignment_wins = sum(r['alignment_tv_improvement_over_dependency_only'] > 0 for r in paired)
    coverage = all(r['missing_teacher_mass_after'] == 0
                   and r['repair_tv_change'] <= config['max_repair_tv_increase']
                   and r['sequence_reconstruction_after'] >= config['min_sequence_reconstruction'] for r in paired)
    transfer = (alignment_mean >= config['min_alignment_tv_improvement']
                and alignment_wins >= config['min_alignment_winning_seeds']
                and coverage)
    return {'confirmation_seeds': seeds, 'known_seed_replications': [s for s in config['seeds'] if s not in seeds],
            'coverage_repair_gate_passed': coverage, 'teacher_alignment_gate_passed': transfer,
            'mean_alignment_tv_improvement_over_dependency_only': alignment_mean,
            'alignment_winning_seeds': alignment_wins, 'paired_confirmation_results': paired,
            'scope': 'New student seeds with the same synthetic teacher/domain; no new-domain claim.'}



def coarse_stability_decision(rows, config):
    selected = [r for r in rows if r['seed'] in config['confirmation_seeds']]
    failures = []
    for row in selected:
        stats = row['coarse_maintenance']
        reason = []
        if row['population_coarse_active_codes'] != config['codebook_size']:
            reason.append('inactive_population_codes')
        if row['population_coarse_perplexity'] < config['min_coarse_perplexity']:
            reason.append('low_population_perplexity')
        if stats['monitored_steps'] == 0 or stats['collapsed_fraction'] > config['max_coarse_collapsed_fraction']:
            reason.append('unstable_hard_usage')
        if row['zero_support_teacher_mass'] > 0 or row['population_sequence_reconstruction'] < config['min_sequence_reconstruction']:
            reason.append('lost_output_or_reconstruction_coverage')
        if reason:
            failures.append({'seed': row['seed'], 'arm': row['arm'], 'reasons': reason})
    return {'passed': not failures, 'confirmation_seeds': config['confirmation_seeds'],
            'evaluated_models': len(selected), 'failures': failures,
            'minimum_population_perplexity': min(r['population_coarse_perplexity'] for r in selected),
            'maximum_monitored_collapsed_fraction': max(r['coarse_maintenance']['collapsed_fraction'] for r in selected)}


def decision(rows, config):
    paired = []
    for seed in config['seeds']:
        base = next(row for row in rows if row['seed'] == seed and row['arm'] == 'alignment')
        dep = next(row for row in rows if row['seed'] == seed and row['arm'] == 'alignment_dependency')
        paired.append({'seed': seed, 'tv_improvement': base['total_variation'] - dep['total_variation'],
                       'tc_reduction': base['hard_conditional_tc_sum'] - dep['hard_conditional_tc_sum'],
                       'reconstruction_change': dep['population_byte_reconstruction'] - base['population_byte_reconstruction'],
                       'reconstruction_floor_met': dep['population_byte_reconstruction'] >= config['min_reconstruction']})
    mean_tv = float(np.mean([r['tv_improvement'] for r in paired]))
    mean_tc = float(np.mean([r['tc_reduction'] for r in paired]))
    fidelity = all(r['reconstruction_floor_met'] and r['reconstruction_change'] >= -config['max_reconstruction_drop'] for r in paired)
    passed = fidelity and mean_tv >= config['min_tv_improvement'] and mean_tc >= config['min_tc_reduction']
    missing_support = [r['seed'] for r in rows if r['arm'] == 'alignment_dependency' and r['zero_support_teacher_mass'] > 0]
    return {'pilot_gate_passed': passed, 'fidelity_gate_passed': fidelity,
            'dependency_seeds_with_missing_teacher_support': missing_support,
            'all_seeds_tv_improved': all(r['tv_improvement'] > 0 for r in paired),
            'mean_tv_improvement': mean_tv, 'mean_tc_reduction': mean_tc, 'paired_seed_results': paired,
            'interpretation': ('Pilot gates passed; larger independent experiments still required.' if passed else
                               'Pilot did not meet the fixed gates; no SCALE benefit established.')}


def run(config, output, resume=False):
    cfg = dict(config)
    lengths = tuple(cfg['lengths'])
    experiment_rvq_config(cfg)
    TransformerConfig(cfg['width'], cfg['heads'], cfg['layers'])
    codewords(lengths, cfg['codebook_size'])
    codewords((lengths[-1],), 2)
    if cfg['dim'] > cfg['teacher_width'] or cfg['teacher_width'] % 2:
        raise ValueError('Teacher width must be even and >= canonical dimension')
    for field in ('teacher_steps', 'codec_steps', 'prior_steps', 'batch_size', 'checkpoint_every',
                  'train_samples', 'validation_samples', 'calibration_samples'):
        if cfg[field] < 1:
            raise ValueError(f'{field} must be positive')
    if not 0 <= cfg['source_noise'] <= 1 or cfg['soft_temperature'] <= 0:
        raise ValueError('Invalid source noise or soft temperature')
    arms = tuple(cfg.get('arms', ARMS))
    if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(ALL_ARMS):
        raise ValueError('Provide distinct supported arms')
    if not set(ARMS).issubset(arms):
        raise ValueError('The original three controls are required')
    if cfg.get('decoder_repair_steps', 0) < 0:
        raise ValueError('decoder_repair_steps must be nonnegative')
    if 'confirmation_seeds' in cfg:
        if (not cfg['confirmation_seeds'] or len(set(cfg['confirmation_seeds'])) != len(cfg['confirmation_seeds'])
                or not set(cfg['confirmation_seeds']).issubset(cfg['seeds'])):
            raise ValueError('Confirmation seeds must be included in seeds')
        if set(arms) != set(ALL_ARMS) or cfg.get('decoder_repair_steps', 0) < 1:
            raise ValueError('Follow-up requires all four arms and decoder repair')
        if (cfg['decoder_repair_learning_rate'] <= 0 or not 0 <= cfg['min_sequence_reconstruction'] <= 1
                or cfg['max_repair_tv_increase'] < 0
                or not 1 <= cfg['min_alignment_winning_seeds'] <= len(cfg['confirmation_seeds'])):
            raise ValueError('Invalid follow-up thresholds or repair learning rate')
    if 'min_coarse_perplexity' in cfg:
        coarse = experiment_rvq_config(cfg).coarse
        if not coarse.enabled or 'confirmation_seeds' not in cfg:
            raise ValueError('Stability gates require enabled coarse maintenance and confirmation seeds')
        if not 1 <= cfg['min_coarse_perplexity'] <= cfg['codebook_size'] or not 0 <= cfg['max_coarse_collapsed_fraction'] <= 1:
            raise ValueError('Invalid coarse stability thresholds')
        if cfg['codec_steps'] <= max(coarse.temp_anneal_steps, coarse.beta_warmup_steps, coarse.dead_code_grace_steps):
            raise ValueError('Stability experiments need a post-annealing observation window')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Provide distinct student seeds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity = dict(config=cfg, implementation_sha256=sources,
                    versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy')})
    manifest = output / 'protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Protocol or implementation changed; use a new output directory')
        if not resume:
            raise ValueError('Existing experiment: use --resume or a new directory')
    # Persist gates and controls before any teacher/student training or evaluation.
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    snapshot = output / 'source_snapshot' / 'src' / 'ncm'
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(__file__).parent.glob('*.py')):
        (snapshot / source.name).write_bytes(source.read_bytes())
    project = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    if project.exists():
        (output / 'source_snapshot' / 'pyproject.toml').write_bytes(project.read_bytes())
    (output / 'source_snapshot' / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    tokens, p, canonical, draws, teacher_info = teacher_setup(cfg, output, resume)
    codec_candidates, tc_weights = codec_training_distribution(
        draws['train'], len(tokens), cfg.get('codec_sampling', 'empirical'))
    model = NCM(experiment_rvq_config(cfg),
                TransformerConfig(cfg['width'], cfg['heads'], cfg['layers']), 256)
    tokenizer = MultiscaleTokenizer(model.rvq, 256)
    optimizer = optax.adam(cfg['codec_learning_rate'])
    codec_step = jax.jit(make_codec_step(cfg, optimizer))
    results = []
    for seed in cfg['seeds']:
        key = jax.random.key(seed)
        books = init_codebooks(jax.random.fold_in(key, 1), model.rvq)
        first_ids = draws['train'][np.arange(cfg['batch_size']) % len(draws['train'])]
        first = as_batch(tokens[first_ids])
        # Initializations are shared exactly across all arms for this seed.
        params = tokenizer.init(jax.random.fold_in(key, 2), first.tokens, first.mask,
                                first.fallback_bytes, first.fallback_mask, books)['params']
        opt_state = optimizer_state(optimizer.init(params), books, model.rvq, jax.random.fold_in(key, 4))
        initial = TrainState(params, opt_state, books, jnp.asarray(0, jnp.int32))
        initialized_initial = None
        for arm in arms:
            directory = output / f'seed-{seed}' / arm
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / 'result.json'
            if resume and result_path.exists():
                results.append(json.loads(result_path.read_text()))
                continue
            weights = arm_weights(arm, cfg)
            state = initial
            codec_path = directory / 'codec.msgpack'
            if resume and codec_path.exists():
                state = serialization.from_bytes(initial, codec_path.read_bytes())
            elif model.rvq.coarse.enabled:
                if initialized_initial is None:
                    warmups = []
                    for warmup_index in range(model.rvq.coarse.warmup_batches):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, warmup_index, 30]))
                        ids = rng.choice(codec_candidates, cfg['batch_size'])
                        warmups.append(as_batch(tokens[ids]))
                    fitted, coarse = warmup_codebooks(tokenizer, initial.params, initial.codebooks,
                                                      initial.opt_state['coarse'], warmups)
                    initialized_initial = initial.replace(codebooks=fitted, opt_state=dict(initial.opt_state, coarse=coarse))
                state = initialized_initial
                save_state(codec_path, state)  # initialized flag/centroids survive even a step-zero interruption
            ids = first_ids
            # Dynamic weights preserve one identical graph/operation budget for every arm.
            executable = codec_step.lower(initial, first, jnp.asarray(canonical[ids]), weights,
                                          jnp.asarray(tc_weights[ids])).compile()
            codec_flops = executable.cost_analysis().get('flops')
            start = time.perf_counter()
            for index in range(int(state.step), cfg['codec_steps']):
                rng = np.random.default_rng(np.random.SeedSequence([seed, index, 30]))
                ids = rng.choice(codec_candidates, cfg['batch_size'])
                state, metrics = executable(state, as_batch(tokens[ids]), jnp.asarray(canonical[ids]), weights,
                                            jnp.asarray(tc_weights[ids]))
                if (index + 1) % cfg['checkpoint_every'] == 0 or index + 1 == cfg['codec_steps']:
                    save_state(codec_path, state)
                    print(json.dumps({'seed': seed, 'arm': arm, 'stage': 'codec', 'step': index + 1,
                                      **{k: float(v) for k, v in metrics.items()}}), flush=True)
            codec_seconds = time.perf_counter() - start
            coarse_extra = {}
            if model.rvq.coarse.enabled:
                tracking = state.opt_state['coarse']
                monitored = int(tracking.monitored_steps)
                coarse_extra = {'coarse_maintenance': {
                    'initialized': bool(tracking.initialized), 'initial_samples': int(tracking.initial_samples),
                    'step': int(tracking.step), 'hard_ema_counts': np.asarray(tracking.hard_counts).tolist(),
                    'replacements': np.asarray(tracking.replacements).tolist(),
                    'monitored_steps': monitored, 'collapsed_steps': int(tracking.collapsed_steps),
                    'collapsed_fraction': float(tracking.collapsed_steps) / max(monitored, 1),
                    'minimum_monitored_perplexity': float(tracking.min_hard_perplexity),
                    'final_temperature': float(schedules(model.rvq.coarse, tracking)[0]),
                    'final_beta': float(schedules(model.rvq.coarse, tracking)[1])}}
            frozen_hash = digest(state_bytes(state))
            encoded = tokenizer.apply({'params': state.params}, first.tokens, first.mask,
                                      first.fallback_bytes, first.fallback_mask, state.codebooks).rvq
            generator = SharedLevelGenerator(model.rvq, model.transformer)
            prior_params = generator.init(jax.random.fold_in(key, 3), state.codebooks, encoded.mask,
                                          encoded.indices)['params']
            prior_optimizer = optax.adam(cfg['prior_learning_rate'])
            prior = PriorState(prior_params, prior_optimizer.init(prior_params), jnp.asarray(0, jnp.int32))
            prior_path = directory / 'prior.msgpack'
            if resume and prior_path.exists():
                prior = serialization.from_bytes(prior, prior_path.read_bytes())
            prior_step = jax.jit(make_prior_step(model, prior_optimizer, state.params, state.codebooks))
            executable_prior = prior_step.lower(prior, first, key).compile()
            prior_flops = executable_prior.cost_analysis().get('flops')
            start = time.perf_counter()
            for index in range(int(prior.step), cfg['prior_steps']):
                rng = np.random.default_rng(np.random.SeedSequence([seed, index, 31]))
                ids = rng.choice(draws['train'], cfg['batch_size'])
                prior, loss = executable_prior(prior, as_batch(tokens[ids]), jax.random.fold_in(key, index + 1000))
                if (index + 1) % cfg['checkpoint_every'] == 0 or index + 1 == cfg['prior_steps']:
                    save_state(prior_path, prior)
                    print(json.dumps({'seed': seed, 'arm': arm, 'stage': 'prior', 'step': index + 1,
                                      'loss': float(loss)}), flush=True)
            prior_seconds = time.perf_counter() - start
            if digest(state_bytes(state)) != frozen_hash:
                raise ValueError('Codec changed during prior training')
            evaluation = evaluate(cfg, model, state, prior, tokens, p, canonical, draws['validation'])
            extra = {}
            if cfg.get('decoder_repair_steps', 0):
                before = evaluation
                state, repair = repair_decoder(cfg, model, state, tokens, draws['train'], seed, directory, resume)
                evaluation = evaluate(cfg, model, state, prior, tokens, p, canonical, draws['validation'])
                if not np.isclose(before['hard_conditional_tc_sum'], evaluation['hard_conditional_tc_sum'], atol=1e-7):
                    raise ValueError('Decoder repair changed population hard-code dependence')
                extra = {'before_repair': before, 'decoder_repair': repair,
                         'repaired_codec_sha256': digest(state_bytes(state))}
            row = dict(evaluation, **extra, **coarse_extra, seed=seed, arm=arm, codec_sha256=frozen_hash,
                       prior_sha256=digest(state_bytes(prior)),
                       codec_step_flops_estimate=codec_flops, prior_step_flops_estimate=prior_flops,
                       codec_seconds_this_invocation=codec_seconds, prior_seconds_this_invocation=prior_seconds)
            result_path.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            results.append(row)
    flop_pairs = {(row['codec_step_flops_estimate'], row['prior_step_flops_estimate'],
                   row.get('decoder_repair', {}).get('step_flops_estimate')) for row in results}
    verdict = decision(results, cfg)
    matched = (len(flop_pairs) == 1
               and all(r['codec_step_flops_estimate'] is not None and r['prior_step_flops_estimate'] is not None
                       and (not cfg.get('decoder_repair_steps', 0) or r['decoder_repair']['step_flops_estimate'] is not None)
                       for r in results))
    verdict['matched_student_flop_estimates'] = matched
    if not matched:
        verdict['pilot_gate_passed'] = False
        verdict['interpretation'] = 'Student operation budgets differ; matched-budget benefit is not established.'
    followup = followup_decision(results, cfg) if 'confirmation_seeds' in cfg else None
    if followup is not None:
        followup['matched_student_flop_estimates'] = matched
        if not matched:
            followup['teacher_alignment_gate_passed'] = False
            followup['coverage_repair_gate_passed'] = False
    stability = coarse_stability_decision(results, cfg) if 'min_coarse_perplexity' in cfg else None
    if stability is not None:
        stability['matched_student_flop_estimates'] = matched
        stability['passed'] = stability['passed'] and matched
    report = {'coarse_stability_decision': stability, 'followup_decision': followup, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'backend': jax.default_backend(),
              'device': str(jax.devices()[0]), 'scope': 'finite-domain, one-teacher mechanism pilot',
              'teacher': teacher_info, 'protocol': identity, 'results': results, 'decision': verdict,
              'budget': {'nominal_code_bits': sum(lengths) * float(np.log2(cfg['codebook_size'])),
                         'codec_steps_per_arm': cfg['codec_steps'],
                         'codec_sampling': cfg.get('codec_sampling', 'empirical'),
                         'prior_sampling': 'empirical teacher-training observations', 'prior_steps_per_arm': cfg['prior_steps'],
                         'decoder_repair_steps_per_arm': cfg.get('decoder_repair_steps', 0),
                         'batch_size': cfg['batch_size'], 'shared_teacher_steps': cfg['teacher_steps'],
                         'coarse_warmup_batches_per_seed': model.rvq.coarse.warmup_batches if model.rvq.coarse.enabled else 0,
                         'coarse_kmeans_iterations': model.rvq.coarse.kmeans_iterations if model.rvq.coarse.enabled else 0,
                         'note': 'All arms execute teacher alignment and soft TC computations with dynamic coefficients; shared teacher/calibration cost is charged equally. FLOPs are compiler estimates, not measured total hardware work.'},
              'limitations': ['One locally trained synthetic teacher, not an externally pretrained LLM.',
                             'PCA representation calibration, no exact operator transplantation or cross-size transfer.',
                             'Independent draws share the finite support; this is not held-out-domain language evaluation.',
                             'Soft training dependence and exact hard-code dependence are different distributions.',
                             'Enumeration grows exponentially and is capped at 4096 codewords.',
                             'No TPU, language-quality, latency advantage or training-compute saving is established.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Minimal SCALE mechanism pilot', '', verdict['interpretation'], '',
             '| Seed | Arm | Byte reconstruction | Hard conditional TC (nats) | Exact output TV | Missing teacher support |',
             '|---:|---|---:|---:|---:|---:|']
    for row in results:
        lines.append(f"| {row['seed']} | {row['arm']} | {row['population_byte_reconstruction']:.4f} | "
                     f"{row['hard_conditional_tc_sum']:.4f} | {row['total_variation']:.4f} | {row['zero_support_teacher_mass']:.4f} |")
    lines += ['', f"Dependence-aware seeds with missing teacher support (infinite forward KL): {verdict['dependency_seeds_with_missing_teacher_support']}.",
              f"Mean paired TV improvement: {verdict['mean_tv_improvement']:.4f}.",
              f"Mean paired hard-TC reduction: {verdict['mean_tc_reduction']:.4f} nats.",
              f"Fidelity gate passed: {verdict['fidelity_gate_passed']}.",
              f"Matched student FLOP estimates: {verdict['matched_student_flop_estimates']}.", '',
              'Gates and weights were fixed in protocol.json before training. This is a small synthetic',
              'mechanism test, not evidence of pretrained-LLM conversion or general language quality.',
              'See report.json for seed-level results, exact distributions, budgets and limitations.']
    if followup is not None:
        lines += ['', '## Four-arm and coverage follow-up', '',
                  f"Confirmation student seeds: {followup['confirmation_seeds']}.",
                  f"Coverage repair gate passed: {followup['coverage_repair_gate_passed']}.",
                  f"Teacher alignment gate passed: {followup['teacher_alignment_gate_passed']}.",
                  f"Mean alignment TV improvement over dependence only: {followup['mean_alignment_tv_improvement_over_dependency_only']:.4f}.", '',
                  '| New seed | Alignment gain over dependence only | Repair TV change | Missing teacher mass before | After | Sequence reconstruction after |',
                  '|---:|---:|---:|---:|---:|---:|']
        for row in followup['paired_confirmation_results']:
            lines.append(f"| {row['seed']} | {row['alignment_tv_improvement_over_dependency_only']:.4f} | {row['repair_tv_change']:.4f} | "
                         f"{row['missing_teacher_mass_before']:.4f} | {row['missing_teacher_mass_after']:.4f} | {row['sequence_reconstruction_after']:.4f} |")
        lines += ['', 'Repair replay uses observed training sequences only and updates decoder parameters only.',
                  f"All four arms receive identical repair updates. Known-seed replications: {followup['known_seed_replications']};",
                  'only the listed new student seeds determine follow-up gates. The teacher/domain is unchanged.']
    if stability is not None:
        lines += ['', '## Coarse-codebook stability', '',
                  f"Stability gate passed: {stability['passed']}.",
                  f"Minimum population coarse perplexity: {stability['minimum_population_perplexity']:.4f}.",
                  f"Maximum post-annealing collapsed-step fraction: {stability['maximum_monitored_collapsed_fraction']:.4f}.", '',
                  '| Seed | Arm | Coarse codes used | Population perplexity | Collapsed steps / monitored | Replacements |',
                  '|---:|---|---:|---:|---:|---:|']
        for row in results:
            track = row['coarse_maintenance']
            lines.append(f"| {row['seed']} | {row['arm']} | {row['population_coarse_active_codes']} | {row['population_coarse_perplexity']:.4f} | "
                         f"{track['collapsed_steps']} / {track['monitored_steps']} | {sum(track['replacements'])} |")
        lines += ['', 'Full failure reasons and persisted schedules/usage counters are recorded in report.json.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output, resume=args.resume)


if __name__ == '__main__':
    main()
