#!/usr/bin/env python3
"""Real-text quality/compute/latency experiment; all arms are trained locally.

Example:
  python scripts/run_decisive_benchmark.py --train train.jsonl --heldout test.jsonl \
      --output runs/decisive --length 128 --steps 1000 --codec-steps 1000

Use --length 512 for the second bucket. Defaults: five warmups, thirty timed
runs, batches 1/16/128. Small overrides permit CPU smoke tests, not quality claims.
"""
import argparse
import csv
from dataclasses import asdict
from flax import serialization
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ncm.baselines import CausalAR, matched_ar_config, parameter_count, sample_conditioned, shifted_inputs
from ncm.benchmark import quality_summary
from ncm.config import NCMConfig, RVQConfig, TransformerConfig
from ncm.data import load_text
from ncm.generation import encode, level_geometry
from ncm.generator import SharedLevelGenerator
from ncm.grids import align, grid_matrices
from ncm.losses import cross_entropy
from ncm.speculative import DistributionAdapter, speculative_generate
from ncm.tokenizer import MultiscaleTokenizer
from ncm.training import Batch, NCM, initialize, make_train_step


def compiler_flops(lowered):
    analysis = lowered.cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0]
    return analysis.get('flops') if analysis else None


def measure(fn, args, warmups, runs, collect=None):
    lowered = jax.jit(fn).lower(*args)
    executable = lowered.compile()
    keys = list(jax.random.split(args[-1], warmups + runs))
    jax.block_until_ready(keys)
    for rng in keys[:warmups]:
        jax.block_until_ready(executable(*args[:-1], rng))
    times = []
    observations = []
    for rng in keys[warmups:]:
        start = time.perf_counter()
        result = executable(*args[:-1], rng)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
        if collect is not None:observations.append(collect(result))
    return dict(compiler_flops=compiler_flops(lowered), median_ms=float(np.median(times)*1000),
                latency_seconds=times, **({'observations':observations} if collect is not None else {}))


def batch(tokens, mask):
    return Batch(tokens, mask, jnp.zeros_like(tokens), jnp.zeros_like(mask))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train = load_text(args.train, args.length + args.prompt_len, expected_sha256=args.train_sha256)
    held = load_text(args.heldout, args.length + args.prompt_len, expected_sha256=args.heldout_sha256)
    # Document identity and exact content leakage, including plain-text files.
    def contents(path):
        path = Path(path)
        if path.suffix == '.jsonl':
            docs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            return {d['id'] for d in docs}, {hashlib.sha256(d['text'].encode()).hexdigest() for d in docs}
        return set(), {hashlib.sha256(path.read_bytes()).hexdigest()}
    a, b = contents(args.train), contents(args.heldout)
    if a[0] & b[0] or a[1] & b[1]:
        raise ValueError('train/heldout document leakage')
    # Prompts must be entirely observed; no padded prompt is passed to the AR cache.
    def usable(data):
        keep = data.mask[:, :args.prompt_len+1].all(1)
        if not keep.any():
            raise ValueError('no documents with a full prompt and continuation')
        return jnp.asarray(data.tokens[keep]), jnp.asarray(data.mask[keep])
    tx, tm = usable(train)
    hx, hm = usable(held)
    p, length = args.prompt_len, args.length
    key = jax.random.key(args.seed)
    transform = TransformerConfig(args.width, args.heads, args.layers)
    rvq = RVQConfig((max(1, length//16), max(1, length//4), length), args.codes, args.dim)
    codec_model = NCM(rvq, transform, 256)
    optimizer = optax.adam(args.learning_rate)
    first = batch(tx[:1, p:], tm[:1, p:])
    state = initialize(codec_model, optimizer, key, first)
    raw_codec_step = make_train_step(codec_model, optimizer)
    codec_step = jax.jit(raw_codec_step)
    def training_batch(i):
        ids = jax.random.randint(jax.random.fold_in(key, i+10), (args.train_batch,), 0, len(tx))
        return tx[ids], tm[ids]
    x, mask = training_batch(0)
    codec_flops = compiler_flops(codec_step.lower(state, batch(x[:, p:], mask[:, p:]),
                                                 jnp.asarray(0), key))
    for i in range(args.codec_steps):
        x, mask = training_batch(i)
        state, _ = codec_step(state, batch(x[:, p:], mask[:, p:]), jnp.asarray(0), jax.random.fold_in(key, i))
    jax.block_until_ready(state)
    cp, books = state.params['MultiscaleTokenizer_0'], state.codebooks
    codec = MultiscaleTokenizer(rvq, 256)
    records, quality, sizes = [], {}, {}
    teacher = CausalAR(TransformerConfig(args.width*2, args.heads, args.layers*2), p+length+args.draft_len+1)
    teacher_params = teacher.init(key, shifted_inputs(tx[:1]), tm[:1])['params']

    def fit_ar(model, params, distilled=False):
        opt_state = optimizer.init(params)
        def step(params, opt_state, x, mask):
            def objective(params):
                logits = model.apply({'params': params}, shifted_inputs(x), mask)
                selected = mask & (jnp.arange(x.shape[1])[None] >= p)
                ce = cross_entropy(logits, x, selected)
                if distilled:
                    target = jax.nn.softmax(teacher.apply({'params': teacher_params}, shifted_inputs(x), mask), -1)
                    kl = jnp.sum(target * (jnp.log(jnp.maximum(target, 1e-30)) - jax.nn.log_softmax(logits)), -1)
                    ce = .5*ce + .5*jnp.sum(jnp.where(selected, kl, 0))/jnp.maximum(selected.sum(), 1)
                return ce
            loss, grads = jax.value_and_grad(objective)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss
        x, mask = training_batch(0)
        flops = compiler_flops(jax.jit(step).lower(params, opt_state, x, mask))
        step = jax.jit(step)
        for i in range(args.steps):
            x, mask = training_batch(i)
            params, opt_state, _ = step(params, opt_state, x, mask)
        jax.block_until_ready(params)
        return params, flops
    teacher_params, teacher_flops = fit_ar(teacher, teacher_params)
    prior_models = {}
    for coarse in (0, 1):
        gen = SharedLevelGenerator(rvq, transform, NCMConfig(coarse, p))
        target = encode(codec_model, cp, books, first)
        gp = gen.init(key, books, target.mask, target.indices, prompt_ids=tx[:1, :p])['params']
        prior_models[coarse] = gen, gp
    # Match the small AR to deployed NCM total parameters (codec + prior).
    codebook_parameters = parameter_count(books.embeddings)
    target_size = parameter_count(cp) + parameter_count(prior_models[1][1]) + codebook_parameters
    ar_config, _ = matched_ar_config(target_size, heads=args.heads, layers=args.layers,
                                    max_length=teacher.max_length)
    small = CausalAR(ar_config, teacher.max_length)
    initial = small.init(key, shifted_inputs(tx[:1]), tm[:1])['params']
    def evaluate_ar(model, params):
        total = count = 0.
        evaluate = jax.jit(lambda x, mask: model.apply({'params': params}, shifted_inputs(x), mask))
        for offset in range(0, len(hx), args.train_batch):
            x, mask = hx[offset:offset+args.train_batch], hm[offset:offset+args.train_batch]
            selected = mask & (jnp.arange(x.shape[1])[None] >= p)
            logits = evaluate(x, mask)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, x)
            total += float(jnp.where(selected, nll, 0).sum())
            count += int(selected.sum())
        return total/count/np.log(2.)
    (out/'codec.msgpack').write_bytes(serialization.to_bytes({'params': cp, 'codebooks': books}))
    (out/'teacher.msgpack').write_bytes(serialization.to_bytes(teacher_params))
    samplers = {}
    for name, distilled in [('ar_scratch', False), ('ar_distilled', True)]:
        params, flops = fit_ar(small, initial, distilled)
        bpb = evaluate_ar(small, params)
        (out/(name+'.msgpack')).write_bytes(serialization.to_bytes(params))
        samplers[name] = lambda prompt, rng, params=params: sample_conditioned(small, params, prompt, length, rng)
        sizes[name] = parameter_count(params)
        quality[name] = dict(heldout_bpb=bpb, bpb_kind='exact causal likelihood', training_compiler_flops=flops)

    _, up = grid_matrices(rvq)
    def soft_decode(logits, masks):
        representation = jnp.zeros((masks.shape[1], length, rvq.dim))
        for k in range(rvq.levels):
            vectors = jax.nn.softmax(logits[k], -1) @ books.embeddings[k]
            values, _ = align(vectors, masks[k], up[k])
            representation += values
        return codec.apply({'params': cp}, representation, masks[-1], method=codec.decode)

    def propose(gen, gp, prompt, rng):
        masks = jnp.ones((prompt.shape[0], length), bool)
        geometry = level_geometry(rvq, masks)
        result = gen.apply({'params': gp}, books, geometry, jnp.zeros_like(geometry, jnp.int32),
                           teacher_forcing=False, prompt_ids=prompt, temperature=1., rngs={'sample': rng})
        return codec.apply({'params': cp}, result.representation, masks, method=codec.decode)

    for coarse, (gen, gp) in prior_models.items():
        opt_state = optimizer.init(gp)
        def step(gp, opt_state, x, mask):
            targets = encode(codec_model, cp, books, batch(x[:, p:], mask[:, p:]))
            teacher_logits = teacher.apply({'params': teacher_params}, shifted_inputs(x), mask)[:, p:]
            def objective(gp):
                result = gen.apply({'params': gp}, books, targets.mask, targets.indices, prompt_ids=x[:, :p])
                ce = cross_entropy(result.logits, targets.indices, targets.mask)
                decoded = soft_decode(result.logits, targets.mask)
                probabilities = jax.nn.softmax(teacher_logits, -1)
                kl = jnp.sum(probabilities * (jax.nn.log_softmax(teacher_logits) - jax.nn.log_softmax(decoded)), -1)
                return ce + args.kl_weight * jnp.sum(jnp.where(mask[:, p:], kl, 0))/jnp.maximum(mask[:, p:].sum(), 1)
            loss, grads = jax.value_and_grad(objective)(gp)
            updates, opt_state = optimizer.update(grads, opt_state, gp)
            return optax.apply_updates(gp, updates), opt_state, loss
        x, mask = training_batch(0)
        flops = compiler_flops(jax.jit(step).lower(gp, opt_state, x, mask))
        compiled = jax.jit(step)
        for i in range(args.steps):
            x, mask = training_batch(i)
            gp, opt_state, _ = compiled(gp, opt_state, x, mask)
        prior_models[coarse] = gen, gp
        total = count = 0.
        @jax.jit
        def evaluate_ncm(x, mask):
            targets = encode(codec_model, cp, books, batch(x[:, p:], mask[:, p:]))
            result = gen.apply({'params': gp}, books, targets.mask, targets.indices, prompt_ids=x[:, :p])
            decoded = codec.apply({'params': cp}, result.representation, mask[:, p:], method=codec.decode)
            code_nll = optax.softmax_cross_entropy_with_integer_labels(result.logits, jnp.maximum(targets.indices, 0))
            token_nll = optax.softmax_cross_entropy_with_integer_labels(decoded, x[:, p:])
            return (jnp.where(targets.mask, code_nll, 0).sum()
                    + jnp.where(mask[:, p:], token_nll, 0).sum()), mask[:, p:].sum()
        for offset in range(0, len(hx), args.train_batch):
            nll, nbytes = evaluate_ncm(hx[offset:offset+args.train_batch], hm[offset:offset+args.train_batch])
            total += float(nll)
            count += int(nbytes)
        bound = total/count/np.log(2.)
        name = f'ncm_coarse_{coarse}'
        sizes[name] = parameter_count(cp) + parameter_count(gp) + codebook_parameters
        (out/(name+'.msgpack')).write_bytes(serialization.to_bytes(gp))
        quality[name] = dict(heldout_bpb=float(bound), bpb_kind='joint-code variational upper bound; not exact marginal BPB',
                             training_compiler_flops=flops)
        def sample(prompt, rng, gen=gen, gp=gp):
            latent_key, token_key = jax.random.split(rng)
            return jax.random.categorical(token_key, propose(gen, gp, prompt, latent_key)).astype(jnp.int32)
        samplers[name] = sample

    gen, gp = prior_models[1]
    def draft_apply(params, buffer, position, draft_len, rng):
        prompt = jax.lax.dynamic_slice(buffer, (position-p,), (p,))[None]
        return propose(gen, params, prompt, rng)[0, :draft_len]
    def teacher_apply(params, buffer, position, draft_len):
        # One parallel causal pass supplies every proposal conditional and bonus.
        inputs = jnp.concatenate([jnp.asarray([256], jnp.int32), buffer[:-1]])[None]
        valid = jnp.arange(buffer.shape[0])[None] <= position+draft_len
        logits = teacher.apply({'params': params}, inputs, valid)[0]
        return jax.lax.dynamic_slice(logits, (position, 0), (draft_len+1, 256))
    draft_adapter = DistributionAdapter(gp, draft_apply)
    teacher_adapter = DistributionAdapter(teacher_params, teacher_apply)
    def spec(prompt, rng):
        keys = jax.random.split(rng, len(prompt))
        return jax.vmap(lambda row, k: speculative_generate(draft_adapter, teacher_adapter, row,
                          args.draft_len, length, k))(prompt, keys)
    samplers['ncm_speculative'] = lambda prompt, rng: spec(prompt, rng).tokens
    sizes['ncm_speculative'] = sizes['ncm_coarse_1'] + parameter_count(teacher_params)
    spec_stats = jax.jit(spec)(hx[:args.quality_samples, :p], key)
    quality['ncm_speculative'] = dict(heldout_bpb=evaluate_ar(teacher, teacher_params),
        bpb_kind='exact teacher distribution', acceptance_rate=float(spec_stats.acceptance_rate.mean()),
        tokens_per_forward_pass=float(spec_stats.tokens_per_forward_pass.mean()), training_compiler_flops=None)
    for name, sampler in samplers.items():
        samples = jax.jit(sampler)(hx[:args.quality_samples, :p], key)
        quality[name]['free_running'] = quality_summary(samples, (np.asarray(hx[:, p:]), np.asarray(hm[:, p:])))
        for size in args.batch_sizes:
            prompt = hx[jnp.arange(size) % len(hx), :p]
            result = measure(sampler, (prompt, key), args.warmups, args.runs)
            records.append(dict(arm=name, batch_size=size, length=length, **result,
                **({'latency_note':'pessimistic: no coarse KV cache'} if name in ('ncm_coarse_1','ncm_speculative') else {})))
    teacher_reference = []
    def direct_teacher(prompt, rng):
        return sample_conditioned(teacher, teacher_params, prompt, length, rng)
    for size in args.batch_sizes:
        prompt = hx[jnp.arange(size) % len(hx), :p]
        reference = measure(direct_teacher, (prompt, key), args.warmups, args.runs)
        teacher_reference.append(dict(arm='teacher_reference', batch_size=size, length=length, **reference))
        for row in records:
            if row['arm'] == 'ncm_speculative' and row['batch_size'] == size:
                row['speedup_vs_cached_teacher'] = reference['median_ms']/row['median_ms']
    report = dict(config=vars(args), backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        versions={name: importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax','numpy')},
        implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sorted(Path(__file__).resolve().parents[1].joinpath('src/ncm').glob('*.py'))]},
        train_sha256=train.sha256, heldout_sha256=held.sha256, quality=quality, latency=records, teacher_reference_latency=teacher_reference,
        parameter_counts=sizes, parameter_match_target=target_size,
        ncm_parameter_components=dict(codec=parameter_count(cp), codebook_embeddings=codebook_parameters,
            parallel_prior=parameter_count(prior_models[0][1]), hybrid_prior=parameter_count(prior_models[1][1])),
        architecture=dict(rvq=asdict(rvq), ncm_transformer=asdict(transform), small_ar=asdict(ar_config),
                          teacher=asdict(teacher.config), ar_max_length=teacher.max_length),
        codec_training_compiler_flops=codec_flops,
        training_steps_per_arm=args.steps, codec_training_steps=args.codec_steps,
        parameter_mismatch_fraction={n: (v-target_size)/target_size for n,v in sizes.items()},
        teacher_parameters=parameter_count(teacher_params), teacher_training_compiler_flops=teacher_flops,
        limitations=['Compiler FLOPs are XLA static cost estimates; loops are not a runtime FLOP counter.',
            'NCM BPB is an upper bound; AR BPB is exact. Do not rank these as equal likelihood estimates.',
            'Free-running bigram TV, UTF-8 validity and distinctness are descriptive quality proxies.',
            'Decoded-logit KL uses teacher-forced code contexts and expected embeddings, a training surrogate.',
            'Speculation recomputes full teacher prefixes; latency includes this cost.',
            'Proposal prompt uses the most recent fixed prompt bucket; teacher uses the complete prefix.'])
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    with (out/'results.csv').open('w', newline='') as f:
        fields = ['arm','batch_size','length','median_ms','compiler_flops','heldout_bpb','bpb_kind',
                  'training_compiler_flops','acceptance_rate','tokens_per_forward_pass',
                  'distinct_fraction','valid_utf8_fraction','byte_bigram_tv_to_heldout','speedup_vs_cached_teacher']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        reference_quality = dict(heldout_bpb=quality['ncm_speculative']['heldout_bpb'],
                                 bpb_kind='exact teacher distribution', training_compiler_flops=teacher_flops)
        for row in records + teacher_reference:
            metrics = reference_quality if row['arm'] == 'teacher_reference' else quality[row['arm']]
            writer.writerow({**{k:v for k,v in row.items() if k in fields},
                             **{k:v for k,v in metrics.items() if k in fields},
                             **{k:v for k,v in metrics.get('free_running', {}).items() if k in fields}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'heldout', 'output'):
        parser.add_argument('--'+name, required=True)
    for name in ('train-sha256','heldout-sha256'):
        parser.add_argument('--'+name)
    for name, default in [('length',128),('prompt-len',16),('steps',100),('codec-steps',100),
        ('width',64),('heads',4),('layers',2),('codes',64),('dim',32),('draft-len',8),
        ('train-batch',16),('quality-samples',16),('warmups',5),('runs',30),('seed',0)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1,16,128])
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--kl-weight', type=float, default=1.)
    args = parser.parse_args()
    if (args.length < 1 or args.prompt_len < 0 or not 1 <= args.draft_len <= args.length
            or min(args.steps,args.codec_steps,args.train_batch,args.quality_samples,args.warmups,args.runs,*args.batch_sizes) < 1):
        parser.error('lengths, steps, batches and timing counts must be positive; draft must fit bucket')
    run(args)


if __name__ == '__main__':
    main()
