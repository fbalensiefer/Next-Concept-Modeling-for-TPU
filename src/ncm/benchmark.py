"""Synchronized single-device NCM versus KV-cached AR benchmark.

Local CPU results cannot establish TPU performance or quality-matched speedups.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import LossConfig, NCM, RVQConfig, TransformerConfig, initialize
from .autoregressive import AutoregressiveBytes, sample_ar, shifted_inputs
from .experiment import batch_at, dataset
from .fast_generation import generate_fast
from .generation import PriorState, bytes_to_text, generate
from .losses import cross_entropy
from .prior_experiment import load_codec


def tree_size(tree):
    leaves = jax.tree.leaves(tree)
    return {'elements': sum(x.size for x in leaves), 'bytes': sum(x.nbytes for x in leaves)}


def timed_case(function, arguments, *, byte_count, warmup, repeats, key, trace_dir=None):
    if warmup < 1 or repeats < 2:
        raise ValueError('Use at least one warmup and two timed repetitions')
    arguments = jax.device_put(arguments)
    jax.block_until_ready(arguments)
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments, key)
    lower_seconds = time.perf_counter() - start
    start = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - start
    keys = list(jax.random.split(key, warmup + repeats))
    jax.block_until_ready(keys)
    for subkey in keys[:warmup]:
        jax.block_until_ready(executable(*arguments, subkey))
    timings = []
    for subkey in keys[warmup:]:
        start = time.perf_counter()
        jax.block_until_ready(executable(*arguments, subkey))
        timings.append(time.perf_counter() - start)
    stats = executable.memory_analysis()
    memory = None if stats is None else {
        name: int(getattr(stats, name)) for name in (
            'argument_size_in_bytes', 'output_size_in_bytes',
            'temp_size_in_bytes', 'alias_size_in_bytes')}
    if memory is not None:
        memory['estimated_live_buffer_bytes'] = (
            memory['argument_size_in_bytes'] + memory['output_size_in_bytes']
            + memory['temp_size_in_bytes'] - memory['alias_size_in_bytes'])
    result = {'trace_and_lower_seconds': lower_seconds, 'compile_seconds': compile_seconds,
              'median_ms': float(np.median(timings) * 1000),
              'p90_ms': float(np.percentile(timings, 90) * 1000),
              'min_ms': min(timings) * 1000,
              'bytes_per_second': float(byte_count / np.median(timings)),
              'latency_seconds': timings, 'compiler_memory': memory,
              'cost_analysis': executable.cost_analysis()}
    if trace_dir is not None:
        try:
            with jax.profiler.trace(str(trace_dir)):
                for i in range(3):
                    with jax.profiler.TraceAnnotation('generation', step_num=i):
                        jax.block_until_ready(executable(*arguments, keys[i % len(keys)]))
            result['trace_directory'] = str(trace_dir)
        except (ImportError, RuntimeError) as error:
            result['trace_error'] = str(error)
    return result


def quality_summary(tokens, reference):
    """Descriptive sample statistics, not a semantic-quality benchmark."""
    rows = [bytes(int(x) for x in row if x >= 0) for row in np.asarray(tokens)]
    target, valid = reference[:2]
    target_pairs = (target[:, :-1] * 256 + target[:, 1:])[valid[:, :-1] & valid[:, 1:]]
    pairs = np.concatenate([np.frombuffer(row, np.uint8).astype(np.int32)[:-1] * 256
                            + np.frombuffer(row, np.uint8).astype(np.int32)[1:] for row in rows])
    def distribution(values):
        counts = np.bincount(values, minlength=65536).astype(np.float64)
        return counts / max(counts.sum(), 1)
    valid_utf8 = 0
    for row in rows:
        try:
            row.decode('utf-8')
            valid_utf8 += 1
        except UnicodeDecodeError:
            pass
    return {'distinct_fraction': len(set(rows)) / len(rows),
            'valid_utf8_fraction': valid_utf8 / len(rows),
            'byte_bigram_tv_to_heldout': (float(np.abs(distribution(pairs) - distribution(target_pairs)).sum() / 2)
                                        if len(pairs) and len(target_pairs) else None),
            'sample_text': bytes_to_text(tokens), 'sample_bytes': np.asarray(tokens).tolist()}


def train_ar(config, source, train, valid, output):
    model = AutoregressiveBytes(TransformerConfig(
        config.get('ar_width', source['width']), config.get('ar_heads', source['heads']),
        config.get('ar_layers', source['layers'])), source['lengths'][-1])
    batch = batch_at(train, np.arange(config['ar_batch_size']) % len(train[0]))
    params = model.init(jax.random.key(config['seed']), shifted_inputs(batch.tokens), batch.mask)['params']
    optimizer = optax.adam(config['ar_learning_rate'])
    state = PriorState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))
    checkpoint = output / 'ar_checkpoint.msgpack'
    if checkpoint.exists():
        state = serialization.from_bytes(state, checkpoint.read_bytes())

    @jax.jit
    def step(state, batch):
        def objective(params):
            logits = model.apply({'params': params}, shifted_inputs(batch.tokens), batch.mask)
            return cross_entropy(logits, batch.tokens, batch.mask)
        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        return PriorState(optax.apply_updates(state.params, updates), opt_state, state.step + 1), loss

    start = time.perf_counter()
    resumed_from = int(state.step)
    for index in range(resumed_from, config['ar_steps']):
        indices = np.asarray(jax.random.randint(jax.random.fold_in(jax.random.key(config['seed']), index + 1),
                                               (config['ar_batch_size'],), 0, len(train[0])))
        state, _ = step(state, batch_at(train, indices))
        if (index + 1) % 100 == 0 or index + 1 == config['ar_steps']:
            jax.block_until_ready(state)
            temporary = output / 'ar_checkpoint.tmp'
            temporary.write_bytes(serialization.to_bytes(state))
            temporary.replace(checkpoint)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start

    @jax.jit
    def evaluate(params, batch):
        logits = model.apply({'params': params}, shifted_inputs(batch.tokens), batch.mask)
        return cross_entropy(logits, batch.tokens, batch.mask), batch.mask.sum()
    metrics = {}
    for name, data in [('train', train), ('validation', valid)]:
        numerator = denominator = 0.
        for start in range(0, len(data[0]), config['ar_batch_size']):
            batch = batch_at(data, np.arange(start, min(start + config['ar_batch_size'], len(data[0]))))
            ce, count = map(float, evaluate(state.params, batch))
            numerator += ce * count
            denominator += count
        metrics[name + '_next_byte_ce'] = numerator / denominator
    return model, state.params, dict(metrics, resumed_from=resumed_from, final_step=int(state.step),
                                    training_seconds_this_invocation=elapsed)


def benchmark_models(ncm, codec_params, prior_params, books, ar, ar_params, config, output, trace=False):
    results = []
    for batch_size in config['batch_sizes']:
        length = ncm.rvq.max_length
        for name, function, arguments in [
            ('ncm_padded', lambda cp, pp, cb, key: generate(
                ncm, cp, pp, cb, key, batch_size=batch_size, temperature=config['temperature'])[0],
             (codec_params, prior_params, books)),
            ('ncm_actual_lengths', lambda cp, pp, cb, key: generate_fast(
                ncm, cp, pp, cb, key, batch_size=batch_size, temperature=config['temperature'])[0],
             (codec_params, prior_params, books)),
            ('ar_kv_cache', lambda params, key: sample_ar(
                ar, params, key, batch_size=batch_size, length=length, temperature=config['temperature']),
             (ar_params,))]:
            print(f'Benchmarking {name}: batch={batch_size}, bytes={length}', flush=True)
            record = timed_case(function, arguments, byte_count=batch_size * length,
                                warmup=config['warmup'], repeats=config['repeats'],
                                key=jax.random.key(config['seed']),
                                trace_dir=output / f'trace-{length}-{batch_size}-{name}' if trace else None)
            results.append(dict(record, model=name, batch_size=batch_size, length=length))
    return results


def run(config, output, *, require_backend=None, trace=False):
    cfg = dict(config)
    backend = jax.default_backend()
    if require_backend is not None and backend != require_backend:
        raise ValueError(f'Required {require_backend} backend, found {backend}; no benchmark was run')
    if jax.process_count() != 1:
        raise ValueError('This benchmark supports a single process/device only')
    for field in ('ar_steps', 'ar_batch_size', 'quality_samples'):
        if cfg[field] < 1:
            raise ValueError(f'{field} must be positive')
    if not cfg['batch_sizes'] or any(b < 1 for b in cfg['batch_sizes']):
        raise ValueError('batch_sizes must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, codec, source, codec_hash = load_codec(json.loads(
        (Path(cfg['prior_run']) / 'config.json').read_text())['codec_run'])
    prior_config = json.loads((Path(cfg['prior_run']) / 'config.json').read_text())
    if codec_hash != prior_config['codec_sha256']:
        raise ValueError('Source codec checkpoint changed')
    prior_bytes = (Path(cfg['prior_run']) / 'checkpoint.msgpack').read_bytes()
    prior_params = serialization.msgpack_restore(prior_bytes)['params']
    train = dataset(prior_config.get('train_file', source['train_file']), model.rvq.max_length)
    valid = dataset(prior_config.get('validation_file', source['validation_file']), model.rvq.max_length)
    if train[2] != prior_config['train_sha256'] or valid[2] != prior_config['validation_sha256']:
        raise ValueError('Source corpus changed')
    identity = dict(cfg, codec_sha256=codec_hash,
                    prior_sha256=hashlib.sha256(prior_bytes).hexdigest(),
                    train_sha256=train[2], validation_sha256=valid[2])
    manifest = output / 'config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Benchmark configuration or source data changed; use a new directory')
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    device = jax.devices()[0]
    with jax.default_device(device):
        ar, ar_params, ar_training = train_ar(cfg, source, train, valid, output)
        cp = codec.params['MultiscaleTokenizer_0']
        key = jax.random.key(cfg['seed'])
        greedy_reference = generate(model, cp, prior_params, codec.codebooks, key, batch_size=4, temperature=0.)
        greedy_fast = generate_fast(model, cp, prior_params, codec.codebooks, key, batch_size=4, temperature=0.)
        equivalent = all(np.array_equal(a, b) for a, b in zip(greedy_reference, greedy_fast))
        if not equivalent:
            raise ValueError('Optimized greedy output differs from the reference checkpoint')
        # Finish quality checks before latency measurements; inference is timed separately.
        quality = {}
        for name, fn in [('ncm_padded', generate), ('ncm_actual_lengths', generate_fast)]:
            tokens, _ = jax.jit(lambda k: fn(model, cp, prior_params, codec.codebooks, k,
                                            batch_size=cfg['quality_samples'], temperature=cfg['temperature']))(key)
            quality[name] = quality_summary(tokens, valid)
        tokens = jax.jit(lambda k: sample_ar(ar, ar_params, k, batch_size=cfg['quality_samples'],
                                            length=model.rvq.max_length, temperature=cfg['temperature']))(key)
        quality['ar_kv_cache'] = quality_summary(tokens, valid)
        results = benchmark_models(model, cp, prior_params, codec.codebooks, ar, ar_params, cfg, output, trace)
        synthetic = []
        for length in cfg.get('synthetic_lengths', []):
            if length < 4:
                raise ValueError('synthetic_lengths must be >= 4')
            transform = TransformerConfig(cfg.get('synthetic_width', 64), 4, 2)
            synthetic_model = NCM(RVQConfig((length // 4, length // 2, length), 64, 32), transform, 256)
            zeros = np.zeros((1, length), np.int32)
            batch = batch_at((zeros, np.ones_like(zeros, bool)), np.array([0]))
            state = initialize(synthetic_model, optax.adam(.001), key, batch)
            synthetic_ar = AutoregressiveBytes(transform, length)
            synthetic_ar_params = synthetic_ar.init(key, shifted_inputs(batch.tokens), batch.mask)['params']
            records = benchmark_models(synthetic_model, state.params['MultiscaleTokenizer_0'],
                                       state.params['SharedLevelGenerator_0'], state.codebooks,
                                       synthetic_ar, synthetic_ar_params, cfg, output, trace)
            synthetic.append({'length': length, 'untrained': True, 'width': transform.width,
                              'layers': transform.layers, 'levels': synthetic_model.rvq.lengths,
                              'results': records})
    # Device counters, if available, are process-lifetime counters, not per-case peaks.
    memory_stats = device.memory_stats()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'backend': backend, 'device': str(device), 'device_kind': device.device_kind,
              'scope': 'single-device inference; end-to-end byte generation including NCM decoder and AR cache',
              'platform': platform.platform(), 'python': platform.python_version(),
              'versions': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy')},
              'dtype': 'float32', 'matmul_precision': str(jax.config.jax_default_matmul_precision),
              'source_codec_config': source, 'source_prior_config': prior_config,
              'config': identity, 'ar_training': ar_training,
              'implementation_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(Path(__file__).parent.glob('*.py'))},
              'parameter_sizes': {'ncm_codec': tree_size(cp), 'ncm_prior': tree_size(prior_params),
                                  'ncm_codebook_embeddings': tree_size(codec.codebooks.embeddings),
                                  'ncm_full_ema_state': tree_size(codec.codebooks), 'ar': tree_size(ar_params)},
              'greedy_reference_equivalent': equivalent, 'quality': quality,
              'trained_results': results, 'synthetic_results': synthetic,
              'device_memory_stats_process_lifetime': memory_stats,
              'per_case_measured_peak_device_bytes': None,
              'limitations': ['No TPU results unless backend is tpu.',
                             'Quality and parameter counts are not matched; latency ratios are not quality-matched speedups.',
                             'AR next-byte CE and NCM code CE are different objectives.',
                             'Bigram TV and distinctness are descriptive, not semantic quality.',
                             'Compiler live-buffer estimates are not measured device peak memory.',
                             'Synthetic longer-length models are untrained and have no quality claim.',
                             'Fast and padded sampling use different RNG streams; greedy equivalence is checked.',
                             'No multi-device throughput or multi-host reliability is measured.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Local inference benchmark', '', f'Backend: {backend}; device: {device.device_kind}; float32.', '',
             'These timings do not establish TPU performance or a quality-matched speedup.', '',
             '| Scope | Model | Batch | Bytes/sequence | Median ms | p90 ms | Bytes/s | Compiler live-buffer bytes |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope, records in [('trained tiny fixture', results)] + [
            ('untrained shape check', row['results']) for row in synthetic]:
        for row in records:
            memory = row['compiler_memory']
            lines.append(f"| {scope} | {row['model']} | {row['batch_size']} | {row['length']} | "
                         f"{row['median_ms']:.3f} | {row['p90_ms']:.3f} | {row['bytes_per_second']:.0f} | "
                         f"{memory['estimated_live_buffer_bytes'] if memory else 'unavailable'} |")
    lines += ['', '## Trained fixture sample diagnostics', '',
              'These descriptive statistics do not establish semantic quality or matched quality.', '',
              '| Model | Distinct fraction | Valid UTF-8 fraction | Byte-bigram TV to held-out |',
              '|---|---:|---:|---:|']
    for name, row in quality.items():
        tv = row['byte_bigram_tv_to_heldout']
        lines.append(f"| {name} | {row['distinct_fraction']:.3f} | {row['valid_utf8_fraction']:.3f} | {tv if tv is not None else 'unavailable'} |")
    lines += ['', f"AR held-out next-byte CE: {ar_training['validation_next_byte_ce']:.4f} nats.",
              'This is not comparable to NCM code cross-entropy.', '',
              '## Model storage', '', '| Component | Elements | Bytes |', '|---|---:|---:|']
    for name, size in report['parameter_sizes'].items():
        lines.append(f"| {name} | {size['elements']} | {size['bytes']} |")
    lines += ['', 'Measured per-case peak device memory: unavailable. See report.json for compilation times,',
              'raw latency observations, source hashes, parameter sizes, quality diagnostics and limitations.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-backend', choices=('cpu', 'gpu', 'tpu'))
    parser.add_argument('--trace', action='store_true', help='Capture three separate profiling executions per case')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output,
        require_backend=args.require_backend, trace=args.trace)


if __name__ == '__main__':
    main()
