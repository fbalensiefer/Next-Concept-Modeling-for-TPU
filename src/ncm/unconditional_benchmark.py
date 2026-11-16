"""Bounded unconditional quality/cost comparison for raw, reversible, and RVQ models.

Invocation: ``python -m ncm.unconditional_benchmark --config CONFIG --output DIR``.
All training and selection use train/validation only; the test book is opened
once, after fitting. This is a mechanism pilot, not a multi-document study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from .autoregressive import AutoregressiveBytes, AutoregressiveCategorical, shifted_inputs, sample_ar
from .checkpoint_audit import strict_from_bytes
from .config import LossConfig, TransformerConfig, experiment_rvq_config
from .experiment import batch_at
from .generation import PriorState, encode as encode_rvq, generate as generate_parallel, make_prior_step, level_geometry
from .grids import grid_matrices
from .generator import SharedLevelGenerator
from .tokenizer import MultiscaleTokenizer


def xor_innovations(values: np.ndarray) -> np.ndarray:
    """Bijective first-difference transform over bytes (addition is XOR)."""
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim < 1:
        raise ValueError("input must have a byte axis")
    return np.concatenate([values[..., :1], values[..., 1:] ^ values[..., :-1]], axis=-1)


def xor_inverse(values: np.ndarray) -> np.ndarray:
    """Inverse XOR prefix scan; works independently on each final-axis row."""
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim < 1:
        raise ValueError("input must have a byte axis")
    result = np.empty_like(values)
    result[..., 0] = values[..., 0]
    for i in range(1, values.shape[-1]):
        result[..., i] = result[..., i - 1] ^ values[..., i]
    return result


def xor_inverse_jax(values):
    """JAX associative scan version used for separately timed inverse cost."""
    return jax.lax.associative_scan(jnp.bitwise_xor, values, axis=-1)


def reconstruction_metrics(target: np.ndarray, decoded: np.ndarray) -> dict:
    target, decoded = np.asarray(target), np.asarray(decoded)
    if target.shape != decoded.shape or target.ndim != 2:
        raise ValueError("expected matching [windows,length] arrays")
    matches = target == decoded
    return {"exact_sequence_rate": float(np.all(matches, axis=1).mean()),
            "byte_accuracy": float(matches.mean()), "windows": int(len(target)),
            "bytes": int(target.size)}


def repetition_rate(samples: np.ndarray) -> float:
    values = np.asarray(samples)
    return float(np.mean(values[:, 1:] == values[:, :-1]))


def _sample_provenance(samples: np.ndarray) -> dict:
    values = np.ascontiguousarray(samples)
    return {"count": int(values.shape[0]), "sequence_length": int(values.shape[1]),
            "dtype": str(values.dtype), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def timing_summary(function, *, warmups: int, repeats: int, batch_size: int = 1) -> dict:
    """Compile once, execute/discard warmups, then time synchronized calls."""
    if warmups < 1 or repeats < 2:
        raise ValueError("require >=1 warmup and >=2 timed repetitions")
    start = time.perf_counter()
    executable = jax.jit(function).lower(jax.random.key(0)).compile()
    compile_ms = (time.perf_counter() - start) * 1000
    for i in range(warmups):
        jax.block_until_ready(executable(jax.random.key(i)))
    observations = []
    for i in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(executable(jax.random.key(10000 + i)))
        observations.append((time.perf_counter() - start) * 1000)
    per_seq = np.asarray(observations) / batch_size
    return {"compile_time_ms": compile_ms, "warmup_calls_excluded": warmups,
            "steady_state_ms_per_seq": {"mean": float(np.mean(per_seq)),
                "p50": float(np.percentile(per_seq, 50)),
                "p95": float(np.percentile(per_seq, 95))},
            "observations_ms": observations}


def cpu_timing_summary(function, *, warmups: int, repeats: int, batch_size: int = 1) -> dict:
    """Time a non-JAX prior sampler, excluding warmups from reported samples."""
    for _ in range(warmups):
        function()
    observations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        observations.append((time.perf_counter() - start) * 1000)
    per_seq = np.asarray(observations) / batch_size
    return {"compile_time_ms": 0., "warmup_calls_excluded": warmups,
            "steady_state_ms_per_seq": {"mean": float(np.mean(per_seq)),
                "p50": float(np.percentile(per_seq, 50)),
                "p95": float(np.percentile(per_seq, 95))},
            "observations_ms": observations, "timing_backend": "CPU Python sampler"}


def _fit_ar(values, *, width, heads, layers, steps, batch_size, lr, seed, max_len, output):
    model = AutoregressiveBytes(TransformerConfig(width, heads, layers), max_len)
    first = jnp.asarray(values[:batch_size], jnp.int32)
    valid = jnp.ones_like(first, bool)
    params = model.init(jax.random.key(seed), shifted_inputs(first), valid)["params"]
    optimizer = optax.adam(lr)
    state = PriorState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))

    @jax.jit
    def step(state, x):
        def objective(p):
            logits = model.apply({"params": p}, shifted_inputs(x), jnp.ones_like(x, bool))
            labels = optax.softmax_cross_entropy_with_integer_labels(logits, x)
            return labels.mean()
        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, opt = optimizer.update(grads, state.opt_state, state.params)
        return PriorState(optax.apply_updates(state.params, updates), opt, state.step + 1), loss

    started = time.perf_counter()
    for i in range(steps):
        ids = np.asarray(jax.random.randint(jax.random.fold_in(jax.random.key(seed), i),
                                            (batch_size,), 0, len(values)))
        state, _ = step(state, jnp.asarray(values[ids], jnp.int32))
    jax.block_until_ready(state)
    train_s = time.perf_counter() - started
    output.write_bytes(serialization.to_bytes(state))
    return model, state.params, train_s


def _ngram_jsd(reference: np.ndarray, samples: np.ndarray, n: int = 2) -> float:
    def counts(x):
        c = {}
        for row in x:
            for i in range(len(row) - n + 1):
                key = bytes(row[i:i+n])
                c[key] = c.get(key, 0) + 1
        return c
    a, b = counts(reference), counts(samples)
    keys = set(a) | set(b)
    pa = np.array([a.get(k, 0) for k in keys], np.float64); pa /= pa.sum()
    pb = np.array([b.get(k, 0) for k in keys], np.float64); pb /= pb.sum()
    m = (pa + pb) / 2
    def kl(p):
        use = p > 0
        return np.sum(p[use] * np.log2(p[use] / m[use]))
    return float((kl(pa) + kl(pb)) / 2)


def _window_se(values: np.ndarray, length: int) -> dict:
    values = np.asarray(values, np.float64)
    return {"mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "windows": len(values), "document_count": 1,
            "interpretation": "window-level descriptive uncertainty; windows are clustered within one test book"}


def pareto_stop(candidates: dict[str, dict], *, tolerance: dict | None = None) -> dict:
    """Require an RVQ candidate to strictly dominate a lossless control in all metrics."""
    tol = tolerance or {"held_out_bpb": 0., "sample_ngram_jsd": 0., "total_generation_ms_per_seq": 0.}
    controls = ["raw_byte_cached_ar", "xor_reversible_ar"]
    rvq = [name for name in candidates if name.startswith("rvq_")]
    wins = []
    for name in rvq:
        row = candidates[name]
        for control in controls:
            base = candidates[control]
            metrics = ("held_out_bpb", "sample_ngram_jsd", "total_generation_ms_per_seq")
            if all(row.get(m) is not None and base.get(m) is not None and
                   row[m] <= base[m] - tol[m] for m in metrics):
                wins.append({"candidate": name, "dominates": control})
    proceed = bool(wins)
    return {"status": "proceed_to_tpu_scaling" if proceed else "halt_quantizer_expansion_and_pivot",
            "rationale": "At least one RVQ candidate strictly dominates both lossless controls on all three requested frontier metrics." if proceed else
                "No RVQ candidate strictly dominates both lossless controls on held-out BPB, sample bigram JSD, and complete generation latency within the specified measurement margins.",
            "strict_dominance_witnesses": wins, "measurement_tolerance": tol}


class ChowLiuPrior:
    """Smoothed Chow–Liu tree for one discrete code level, fitted on train only."""
    def __init__(self, values: np.ndarray, vocab: int, alpha: float = 0.5):
        x = np.asarray(values, np.int32)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("tree prior requires [examples,positions]")
        self.vocab, self.length, self.alpha = vocab, x.shape[1], alpha
        self.marginals = np.stack([self._smooth(np.bincount(x[:, i], minlength=vocab)) for i in range(x.shape[1])])
        mi = np.zeros((self.length, self.length), np.float64)
        joints = {}
        for i in range(self.length):
            for j in range(i):
                counts = np.zeros((vocab, vocab), np.float64)
                np.add.at(counts, (x[:, i], x[:, j]), 1.)
                p = (counts + alpha / vocab**2) / (len(x) + alpha)
                pi, pj = p.sum(1), p.sum(0)
                mi[i, j] = mi[j, i] = np.sum(p * np.log(np.maximum(p, 1e-15) / np.maximum(pi[:, None] * pj[None, :], 1e-15)))
                joints[(i, j)] = counts
        # Deterministic Prim maximum spanning tree rooted at position zero.
        self.parent = np.full(self.length, -1, np.int32); seen = {0}
        while len(seen) < self.length:
            edge = max(((mi[a, b], a, b) for a in seen for b in range(self.length) if b not in seen), key=lambda z:(z[0],-z[1],-z[2]))
            _, a, b = edge; self.parent[b] = a; seen.add(b)
        self.order = [0]
        while len(self.order) < self.length:
            ready = [i for i in range(1, self.length) if i not in self.order and self.parent[i] in self.order]
            if not ready: raise AssertionError("tree traversal failed")
            self.order.extend(sorted(ready))
        self.conditionals = {}
        for child in range(1, self.length):
            par = int(self.parent[child]); raw = joints[(max(child, par), min(child, par))]
            counts = raw if child > par else raw.T
            self.conditionals[child] = (counts + alpha / vocab) / (counts.sum(1, keepdims=True) + alpha)

    def _smooth(self, counts):
        return (counts + self.alpha / self.vocab) / (counts.sum() + self.alpha)

    def nll(self, values):
        return float(self.nll_rows(values).mean() / self.length)

    def nll_rows(self, values):
        x = np.asarray(values, np.int32); total = -np.log(self.marginals[0, x[:, 0]])
        for child in range(1, self.length):
            total -= np.log(self.conditionals[child][x[:, self.parent[child]], x[:, child]])
        return total

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        out = np.zeros((count, self.length), np.int32)
        for pos in self.order:
            probs = np.broadcast_to(self.marginals[pos], (count, self.vocab)) if pos == 0 else self.conditionals[pos][out[:, self.parent[pos]]]
            cdf = np.cumsum(probs, axis=-1)
            out[:, pos] = np.minimum((rng.random(count)[:, None] > cdf).sum(1), self.vocab - 1)
        return out


def decode_rvq_codes(codes: np.ndarray, model, codec_params, books) -> np.ndarray:
    codes = jnp.asarray(codes, jnp.int32)
    rep = jnp.zeros((codes.shape[0], model.rvq.max_length, model.rvq.dim), jnp.float32)
    _, up = grid_matrices(model.rvq)
    for k, n in enumerate(model.rvq.lengths):
        vectors = jnp.asarray(books.embeddings)[k, codes[:, k, :n]]
        rep += jnp.einsum("ln,bnd->bld", up[k, :, :n], vectors)
    mask = jnp.ones((codes.shape[0], model.rvq.max_length), bool)
    logits = MultiscaleTokenizer(model.rvq, 256).apply({"params": codec_params}, rep, mask,
                                                       method=MultiscaleTokenizer.decode)
    return jnp.argmax(logits, axis=-1)


def run(config: dict, output: Path) -> dict:
    from .text_data import document_dataset, validate_splits
    from .training import Batch, NCM, initialize
    from .fast_generation import generate_fast
    from .prior_training import load_phase1_codec, PriorTrainState, shifted_categorical_inputs

    root = Path(config["root"]).resolve()
    length = int(config.get("length", 512)); seed = int(config.get("seed", 0))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    split_paths = [root / config[k] for k in ("train_file", "validation_file", "test_file")]
    validate_splits(split_paths)
    train, train_info = document_dataset(split_paths[0], length)
    valid, valid_info = document_dataset(split_paths[1], length)
    test, test_info = document_dataset(split_paths[2], length)
    codec_path = root / config["codec_checkpoint"]
    codec_cfg_path = root / config["codec_config"]
    audit = json.loads((root / "reports/audit/checkpoint_restoration_audit.json").read_text())
    audited = next((r for r in audit["checkpoints"] if r["path"] == config["codec_checkpoint"]), None)
    if not audited or audited["status"] != "exact_match" or Path(audited["provenance"]["config"]).name != codec_cfg_path.name:
        raise ValueError("requested codec is not the audited exact_match checkpoint/config pair")
    codec_config = json.loads(codec_cfg_path.read_text())
    if length != codec_config["lengths"][-1]:
        raise ValueError("the verified codec supports only its configured maximum fixed length")
    model, codec_state = load_phase1_codec(codec_config, codec_path, codec_config["seed"])
    tokenizer = MultiscaleTokenizer(model.rvq, 256)
    cp = codec_state.params["MultiscaleTokenizer_0"]
    encoded = []
    for data in (train, valid, test):
        batch = Batch(jnp.asarray(data[0]), jnp.asarray(data[1]), jnp.zeros_like(jnp.asarray(data[0])), jnp.zeros_like(jnp.asarray(data[1])))
        result = tokenizer.apply({"params": cp}, batch.tokens, batch.mask, batch.fallback_bytes, batch.fallback_mask, codec_state.codebooks)
        encoded.append(np.asarray(result.rvq.indices))
    recon_logits = tokenizer.apply({"params": cp}, jnp.asarray(test[0]), jnp.asarray(test[1]),
                                   jnp.zeros_like(jnp.asarray(test[0])), jnp.zeros_like(jnp.asarray(test[1])),
                                   codec_state.codebooks).logits
    codec_decode = np.asarray(recon_logits).argmax(-1)
    fidelity = reconstruction_metrics(test[0], codec_decode)

    width, heads, layers = (int(config[k]) for k in ("width", "heads", "layers"))
    steps, bsize = (int(config[k]) for k in ("training_steps", "batch_size"))
    lr = float(config["learning_rate"])
    byte_candidates = {}
    checkpoint_tmp = Path(tempfile.mkdtemp(prefix="ncm-unconditional-benchmark-"))
    for name, train_values, decode in (("raw_byte_cached_ar", train[0].astype(np.uint8), lambda x: x),
                                       ("xor_reversible_ar", xor_innovations(train[0]), xor_inverse)):
        model_ar, params, train_s = _fit_ar(train_values, width=width, heads=heads, layers=layers,
            steps=steps, batch_size=bsize, lr=lr, seed=seed, max_len=length,
            output=checkpoint_tmp / f"{name}.msgpack")
        eval_values = test[0].astype(np.uint8) if name == "raw_byte_cached_ar" else xor_innovations(test[0])
        logits = model_ar.apply({"params": params}, shifted_inputs(jnp.asarray(eval_values, jnp.int32)), jnp.ones_like(jnp.asarray(eval_values), bool))
        nll = float(optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(eval_values)).mean())
        # Evaluate in transformed coordinates; bijection preserves sequence NLL.
        sampler = lambda key: sample_ar(model_ar, params, key, batch_size=int(config["sample_count"]), length=length, temperature=1.)
        timing = timing_summary(sampler, warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
        sampled_representation = sampler(jax.random.key(seed + 90))
        samples = decode(np.asarray(sampled_representation))
        if name == "xor_reversible_ar":
            inverse_timing = timing_summary(lambda key: xor_inverse_jax(sampled_representation), warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
            decoder_cost = inverse_timing["steady_state_ms_per_seq"]
        else:
            inverse_timing = None
            decoder_cost = {"mean": 0., "p50": 0., "p95": 0.}
        byte_candidates[name] = {"held_out_bpb": nll / np.log(2), "sample_ngram_jsd": _ngram_jsd(test[0], samples),
            "degenerate_adjacent_repetition_rate": repetition_rate(samples),
            "exact_sequence_rate": 1., "training_seconds": train_s, "training_steps": steps,
            "parameter_count": int(sum(x.size for x in jax.tree.leaves(params))),
            "generation_timing": {"prior": timing, "inverse_transform": inverse_timing}, "prior_sampling_ms_per_seq": timing["steady_state_ms_per_seq"],
            "codec_decode_ms_per_seq": decoder_cost,
            "total_generation_ms_per_seq": timing["steady_state_ms_per_seq"]["mean"] + decoder_cost["mean"],
            "sequential_rounds": length, "sample_artifact": _sample_provenance(samples),
            "held_out_window_nll": _window_se(np.asarray(optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(eval_values))).mean(-1)/np.log(2), length)}

    # Reversible transform's exact inverse and stochastic generation are a
    # separate decoder-cost measurement, charged outside the AR prior.
    probe = np.asarray(test[0][:min(32, len(test[0]))], np.uint8)
    if not np.array_equal(xor_inverse(xor_innovations(probe)), probe):
        raise AssertionError("XOR transform failed exact round-trip")

    # Full-AR RVQ prior uses the audited frozen codec and level-major codes.
    prior = AutoregressiveCategorical(TransformerConfig(width, heads, layers), sum(model.rvq.lengths), model.rvq.codebook_size)
    def flatten_codes(indices):
        return jnp.concatenate([jnp.asarray(indices[k, :, :n]) for k,n in enumerate(model.rvq.lengths)], axis=1)
    codes_train = flatten_codes(encoded[0])
    codes_test = flatten_codes(encoded[2])
    pinput = jnp.full((bsize, prior.max_length), prior.vocab_size, jnp.int32)
    pparams = prior.init(jax.random.key(seed + 1), pinput, jnp.ones_like(pinput, bool))["params"]
    p_opt = optax.adam(lr); pstate = PriorTrainState(pparams, p_opt.init(pparams), jnp.asarray(0, jnp.int32))
    @jax.jit
    def ar_step(state, target):
        inp = shifted_categorical_inputs(target, prior.vocab_size)
        def objective(p):
            logits = prior.apply({"params": p}, inp, jnp.ones_like(target, bool))
            return optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()
        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, optstate = p_opt.update(grads, state.opt_state, state.params)
        return PriorTrainState(optax.apply_updates(state.params, updates), optstate, state.step + 1), loss
    started = time.perf_counter()
    for i in range(steps):
        ids = np.asarray(jax.random.randint(jax.random.fold_in(jax.random.key(seed + 2), i), (bsize,), 0, len(train[0])))
        pstate, _ = ar_step(pstate, codes_train[ids])
    jax.block_until_ready(pstate); full_ar_train_s = time.perf_counter() - started
    def ar_nll(params):
        logits = prior.apply({"params": params}, shifted_categorical_inputs(codes_test, prior.vocab_size), jnp.ones_like(codes_test, bool))
        return optax.softmax_cross_entropy_with_integer_labels(logits, codes_test).mean(-1)
    full_code_nll = np.asarray(ar_nll(pstate.params))

    def sample_full_ar(key):
        count = int(config["sample_count"]); cache = prior.init_cache(count)
        def step(carry, pos):
            prev, cache = carry
            logits, cache = prior.apply({"params": pstate.params}, prev, pos, cache, method=prior.decode_step)
            code = jax.random.categorical(jax.random.fold_in(key, pos), logits).astype(jnp.int32)
            return (code, cache), code
        _, sampled = jax.lax.scan(step, (jnp.full((count,), prior.vocab_size), cache), jnp.arange(prior.max_length))
        return sampled.T
    full_ar_samples_codes = sample_full_ar(jax.random.key(seed + 91))
    full_ar_code_timing = timing_summary(sample_full_ar, warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
    def pad_codes(flat):
        arr = np.zeros((len(flat), model.rvq.levels, model.rvq.max_length), np.int32)
        offset = 0
        for k, n in enumerate(model.rvq.lengths):
            arr[:, k, :n] = flat[:, offset:offset+n]; offset += n
        return arr
    def decode_sample_codes(key):
        return decode_rvq_codes(pad_codes(np.asarray(full_ar_samples_codes)), model, cp, codec_state.codebooks)
    full_ar_samples = decode_sample_codes(jax.random.key(seed + 92))
    full_ar_decode_timing = timing_summary(lambda key: decode_rvq_codes(pad_codes(np.asarray(full_ar_samples_codes)), model, cp, codec_state.codebooks),
        warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
    full_ar_exact = reconstruction_metrics(test[0], codec_decode)
    decoder_ce = np.asarray(optax.softmax_cross_entropy_with_integer_labels(recon_logits, jnp.asarray(test[0])).mean(-1))
    full_ar_bpb = float(np.mean(full_code_nll + decoder_ce) / np.log(2))
    full_ar_total = full_ar_code_timing["steady_state_ms_per_seq"]["mean"] + full_ar_decode_timing["steady_state_ms_per_seq"]["mean"]

    # Train the level-parallel prior with the existing frozen-codec objective.
    parallel_generator = SharedLevelGenerator(model.rvq, model.transformer, model.config)
    dummy_mask = jnp.ones((1, length), bool)
    dummy_levels = level_geometry(model.rvq, dummy_mask)
    dummy_targets = jnp.zeros(dummy_levels.shape, jnp.int32)
    # Initialize the prior from scratch so it receives the same data/update
    # budget as byte and full-AR controls; retain only the audited frozen codec.
    parallel_params = parallel_generator.init(jax.random.key(seed + 5), codec_state.codebooks,
        dummy_levels, dummy_targets, teacher_forcing=True, source_mask=dummy_mask)["params"]
    parallel_opt = optax.adam(lr)
    parallel_state = PriorState(parallel_params, parallel_opt.init(parallel_params), jnp.asarray(0, jnp.int32))
    parallel_step = jax.jit(make_prior_step(model, parallel_opt, cp, codec_state.codebooks))
    started = time.perf_counter()
    for i in range(steps):
        ids = np.asarray(jax.random.randint(jax.random.fold_in(jax.random.key(seed + 3), i), (bsize,), 0, len(train[0])))
        parallel_state, _ = parallel_step(parallel_state, batch_at(train, ids), jax.random.fold_in(jax.random.key(seed + 4), i))
    jax.block_until_ready(parallel_state); parallel_train_s = time.perf_counter() - started
    target_test = encode_rvq(model, cp, codec_state.codebooks, batch_at(test, np.arange(len(test[0]))))
    parallel_logits = parallel_generator.apply({"params": parallel_state.params}, codec_state.codebooks,
        target_test.mask, target_test.indices, teacher_forcing=True, source_mask=batch_at(test, np.arange(len(test[0]))).mask).logits
    level_nll = optax.softmax_cross_entropy_with_integer_labels(parallel_logits, jnp.maximum(target_test.indices, 0))
    parallel_nll_by_window = np.asarray(jnp.sum(jnp.where(target_test.mask, level_nll, 0.), axis=(0,2)))
    parallel_code_nats = float(parallel_nll_by_window.mean())
    parallel_param_count = int(sum(x.size for x in jax.tree.leaves(parallel_state.params)))
    def parallel_codes(key):
        count=int(config["sample_count"]); mask=jnp.ones((count,length),bool)
        levels=level_geometry(model.rvq,mask); targets=jnp.zeros(levels.shape,jnp.int32)
        return parallel_generator.apply({"params":parallel_state.params},codec_state.codebooks,levels,targets,
            teacher_forcing=False,source_mask=mask,temperature=1.,rngs={"sample":key})
    def parallel_decode(key):
        gen=parallel_codes(key)
        return tokenizer.apply({"params":cp},gen.representation,jnp.ones((int(config["sample_count"]),length),bool),method=MultiscaleTokenizer.decode).argmax(-1)
    parallel_prior_timing = timing_summary(parallel_codes, warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
    parallel_decode_timing = timing_summary(parallel_decode, warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
    parallel_samples = np.asarray(parallel_decode(jax.random.key(seed + 93)))
    parallel_total = parallel_prior_timing["steady_state_ms_per_seq"]["mean"] + parallel_decode_timing["steady_state_ms_per_seq"]["mean"]
    parallel_fidelity = reconstruction_metrics(test[0], codec_decode)

    # Chow–Liu tree priors are fitted on frozen training codes, independently
    # within each level. This isolates within-level dependence; it omits coarse
    # conditioning by design and is reported as a diagnostic control.
    tree_fit_started = time.perf_counter()
    trees = [ChowLiuPrior(np.asarray(encoded[0][k, :, :n]), model.rvq.codebook_size) for k, n in enumerate(model.rvq.lengths)]
    tree_nll_nats = sum(tree.nll(np.asarray(encoded[2][k, :, :n])) * n for k, (tree,n) in enumerate(zip(trees, model.rvq.lengths)))
    def sample_tree_codes():
        rng = np.random.default_rng(seed + 500)
        out = np.zeros((int(config["sample_count"]), model.rvq.levels, model.rvq.max_length), np.int32)
        for k,(tree,n) in enumerate(zip(trees, model.rvq.lengths)): out[:,k,:n] = tree.sample(rng, int(config["sample_count"]))
        return out
    tree_prior_timing = cpu_timing_summary(sample_tree_codes, warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
    fixed_tree_codes = sample_tree_codes()
    tree_decode_timing = timing_summary(lambda key: decode_rvq_codes(fixed_tree_codes, model, cp, codec_state.codebooks),
        warmups=int(config["warmups"]), repeats=int(config["timing_repeats"]), batch_size=int(config["sample_count"]))
    tree_samples = np.asarray(decode_rvq_codes(fixed_tree_codes, model, cp, codec_state.codebooks))
    tree_total_ms = tree_prior_timing["steady_state_ms_per_seq"]["mean"] + tree_decode_timing["steady_state_ms_per_seq"]["mean"]
    # Plug-in dependence diagnostic is explicitly finite-sample biased; tree gain
    # is the held-out reduction relative to independent position marginals.
    independent_nll = 0.
    for k,n in enumerate(model.rvq.lengths):
        x = np.asarray(encoded[2][k,:,:n])
        for pos in range(n):
            p = np.bincount(np.asarray(encoded[0][k,:,pos]), minlength=model.rvq.codebook_size).astype(float) + .5
            p /= p.sum(); independent_nll -= np.log(p[x[:,pos]]).sum()
    independent_nll /= (len(test[0]) * length)
    tree_fit_s = time.perf_counter() - tree_fit_started
    tree_exact = fidelity["exact_sequence_rate"]
    tree_nll_by_window=sum(tree.nll_rows(np.asarray(encoded[2][k,:,:n])) for k,(tree,n) in enumerate(zip(trees,model.rvq.lengths)))
    tree_bpb_by_window=(tree_nll_by_window/length+decoder_ce)/np.log(2)
    full_ar_bpb_by_window=(full_code_nll*prior.max_length/length+decoder_ce)/np.log(2)
    rvq_candidates = {
      "rvq_level_parallel": {"held_out_bpb": float((parallel_code_nats / length + decoder_ce.mean()) / np.log(2)),
        "held_out_window_nll": _window_se((parallel_nll_by_window/length+decoder_ce)/np.log(2),length),
        "sample_ngram_jsd": _ngram_jsd(test[0], parallel_samples), "degenerate_adjacent_repetition_rate": repetition_rate(parallel_samples), "exact_sequence_rate": parallel_fidelity["exact_sequence_rate"],
        "codec_fidelity": parallel_fidelity, "code_prior_nll_nats_per_sequence": parallel_code_nats,
        "decoder_nll_nats_per_sequence": float(decoder_ce.mean()), "training_seconds": parallel_train_s, "training_steps": steps,
        "parameter_count": parallel_param_count, "generation_timing": {"prior":parallel_prior_timing,"decoder":parallel_decode_timing},
        "prior_sampling_ms_per_seq": parallel_prior_timing["steady_state_ms_per_seq"],
        "codec_decode_ms_per_seq": parallel_decode_timing["steady_state_ms_per_seq"],
        "total_generation_ms_per_seq": parallel_total,
        "sequential_rounds": model.rvq.levels, "sample_artifact": _sample_provenance(parallel_samples)},
      "rvq_chow_liu": {"held_out_bpb": float((tree_nll_nats / length + decoder_ce.mean()) / np.log(2)),
        "held_out_window_nll": _window_se(tree_bpb_by_window,length),
        "sample_ngram_jsd": _ngram_jsd(test[0], tree_samples), "degenerate_adjacent_repetition_rate": repetition_rate(tree_samples), "exact_sequence_rate": tree_exact,
        "codec_fidelity": fidelity, "training_seconds": tree_fit_s, "parameter_count": 0,
        "tree_edges": [tree.parent.tolist() for tree in trees], "independent_prior_nll_nats_per_byte": float(independent_nll),
        "tree_prior_nll_nats_per_sequence": float(tree_nll_nats), "pairwise_dependence_gain_nats_per_byte": float(independent_nll-tree_nll_nats/length),
        "conditioning_summary_gap": "not identifiable from this pilot; Chow-Liu factors omit earlier-level conditioning",
        "generation_timing": {"prior": tree_prior_timing, "decoder": tree_decode_timing}, "prior_sampling_ms_per_seq": tree_prior_timing["steady_state_ms_per_seq"],
        "codec_decode_ms_per_seq": tree_decode_timing["steady_state_ms_per_seq"],
        "total_generation_ms_per_seq": tree_total_ms, "sequential_rounds": model.rvq.levels,
        "sample_artifact": _sample_provenance(tree_samples)},
      "rvq_full_ar": {"held_out_bpb": float((full_code_nll.mean()*prior.max_length/length + decoder_ce.mean())/np.log(2)),
        "held_out_window_nll": _window_se(full_ar_bpb_by_window,length), "sample_ngram_jsd": _ngram_jsd(test[0], full_ar_samples), "degenerate_adjacent_repetition_rate": repetition_rate(full_ar_samples),
        "exact_sequence_rate": full_ar_exact["exact_sequence_rate"], "codec_fidelity": full_ar_exact,
        "code_prior_nll_nats_per_sequence": float(full_code_nll.mean()*prior.max_length), "decoder_nll_nats_per_sequence": float(decoder_ce.sum()),
        "training_seconds": full_ar_train_s, "training_steps": steps, "parameter_count": int(sum(x.size for x in jax.tree.leaves(pstate.params))),
        "generation_timing": {"prior": full_ar_code_timing,"decoder":full_ar_decode_timing},
        "prior_sampling_ms_per_seq": full_ar_code_timing["steady_state_ms_per_seq"],
        "codec_decode_ms_per_seq": full_ar_decode_timing["steady_state_ms_per_seq"],
        "total_generation_ms_per_seq": full_ar_total, "sequential_rounds": prior.max_length, "sample_artifact": _sample_provenance(full_ar_samples)},
    }
    candidates = {**byte_candidates, **rvq_candidates}
    stop = pareto_stop(candidates, tolerance=config.get("pareto_tolerance", {"held_out_bpb": 0.05,"sample_ngram_jsd": 0.005,"total_generation_ms_per_seq": 0.1}))
    best_rvq_name = min((n for n in candidates if n.startswith("rvq_")),
        key=lambda n: candidates[n]["held_out_bpb"])
    best_rvq = candidates[best_rvq_name]
    raw, reversible = candidates["raw_byte_cached_ar"], candidates["xor_reversible_ar"]
    stop["rationale"] += (f" Lowest-BPB RVQ candidate={best_rvq_name} (BPB/bound={best_rvq['held_out_bpb']:.4f}, "
        f"bigram JSD={best_rvq['sample_ngram_jsd']:.4f}, total={best_rvq['total_generation_ms_per_seq']:.3f} ms/sequence); "
        f"raw control=({raw['held_out_bpb']:.4f}, {raw['sample_ngram_jsd']:.4f}, {raw['total_generation_ms_per_seq']:.3f}); "
        f"reversible control=({reversible['held_out_bpb']:.4f}, {reversible['sample_ngram_jsd']:.4f}, {reversible['total_generation_ms_per_seq']:.3f}). "
        f"Frozen codec exact sequence reconstruction={fidelity['exact_sequence_rate']:.6g}, byte accuracy={fidelity['byte_accuracy']:.6g}; "
        f"within-level Chow-Liu gain={((independent_nll-tree_nll_nats/length)/np.log(2)):.4f} bits/byte; "
        f"level count={model.rvq.levels}, full-AR code draws={prior.max_length}, test windows={len(test[0])}.")
    for row in candidates.values():
        if "held_out_window_nll" not in row: row["held_out_window_nll"] = None
    return {"schema_version": 1, "config": config, "backend": jax.default_backend(),
        "git_commit": __import__("subprocess").run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True).stdout.strip(),
        "implementation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "runtime_versions": {"python": platform.python_version(), "jax": jax.__version__,
            "backend": jax.default_backend(), "numpy": np.__version__},
        "split_metadata": {"train": {"document_ids": sorted(set(train_info["document_ids"])), "window_count": len(train[0]), "dropped_tail_bytes": train_info["dropped_tail_bytes"]},
            "validation": {"document_ids": sorted(set(valid_info["document_ids"])), "window_count": len(valid[0]), "dropped_tail_bytes": valid_info["dropped_tail_bytes"]},
            "test": {"document_ids": sorted(set(test_info["document_ids"])), "window_count": len(test[0]), "dropped_tail_bytes": test_info["dropped_tail_bytes"]},
            "file_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest() for label,path in zip(("train","validation","test"),split_paths)},
            "train_document_count": 1, "validation_document_count": 1, "test_document_count": 1,
            "test_document_support_caveat": "One distinct Gutenberg book per split; test windows do not overlap training-book data, but document-level support is one test book. Reported window SE is clustered/descriptive, not an independent-document interval."},
        "codec": {"path": config["codec_checkpoint"], "config": config["codec_config"], "checkpoint_sha256": audited["sha256"],
            "config_sha256": audited["provenance"]["config_sha256"], "strict_status": audited["status"],
            "fidelity": fidelity, "fidelity_window_level": {"exact": _window_se(np.all(test[0] == codec_decode,axis=1), length),
                "byte_accuracy": _window_se((test[0] == codec_decode).mean(axis=1), length)}},
        "candidate_budget": {"steps_per_neural_prior": steps, "batch_size": bsize, "optimizer": "Adam", "learning_rate": lr,
            "width": width, "heads": heads, "layers": layers,
            "initialization": "All neural priors trained from fresh random initialization; only the audited RVQ encoder/decoder is frozen and reused.",
            "budget_note": "Equal update count, batch size, optimizer, learning rate, and transformer width/depth; parameter counts and sequence lengths differ, so total parameter and FLOP budgets are not identical. Training time, parameter counts, and sequential rounds are reported per candidate; treat this as a bounded matched-protocol pilot, not a compute-normalized scaling result."},
        "candidates": candidates, "stop_rule_decision": stop,
        "within_level_dependence_diagnostic": {"kind": "held-out Chow-Liu gain over independent marginals, bits/byte",
            "gain_bits_per_byte": float((independent_nll-tree_nll_nats/length)/np.log(2)),
            "conditioning_summary_gap": "not identified; requires matched conditional vs unconditional factorization experiment"},
        "status": "bounded_unconditional_pilot", "limitations": ["One held-out book limits document-level inference.",
            "Equal steps and architecture dimensions do not equalize actual compute across output vocabularies or autoregressive sequence lengths.",
            "The Chow-Liu diagnostic omits cross-level conditioning; it is not the NCM conditional total correlation.",
            "Window confidence intervals are descriptive and do not replace multiple test documents."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config, args.output)
    (args.output / "benchmark_results.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = result["candidates"]
    lines = ["# Bounded unconditional generation benchmark", "", "All candidates use non-overlapping 512-byte windows from distinct Gutenberg books in train/validation/test. There is one book per split; window-level uncertainty is descriptive and clustered within the single test book.", "", "| Candidate | Held-out BPB/bound | Bigram JSD | Exact codec rate | Total ms/sequence | Sequential rounds |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row.get('held_out_bpb', float('nan')):.4f} | {row.get('sample_ngram_jsd', float('nan')):.4f} | {row.get('exact_sequence_rate', float('nan')):.4f} | {row.get('total_generation_ms_per_seq', float('nan')):.3f} | {row.get('sequential_rounds', '—')} |")
    fidelity = result["codec"]["fidelity"]
    lines += ["", f"Frozen RVQ reconstruction exact-window rate: {fidelity['exact_sequence_rate']:.6f}; byte accuracy: {fidelity['byte_accuracy']:.6f}.", "", "Mathematically, the RVQ bound is (NLL of level-major codes under the fitted prior + frozen decoder cross-entropy on the held-out bytes) / (byte count × ln 2). It upper-bounds the codec model's held-out byte NLL; it includes both prior and reconstruction error. Raw and reversible candidates report their fitted autoregressive byte-chain NLL in bits/byte, with exact reconstruction by the bijective transform. The codec term breakdown is in JSON.", "", "Uncertainty is summarized over 66 non-overlapping test windows from one held-out book. Window standard errors describe this book only and do not estimate between-document variation.", "", f"Stop rule: `{result['stop_rule_decision']['status']}`. {result['stop_rule_decision']['rationale']}", "", "The Chow–Liu tree is a within-level diagnostic and omits earlier-level conditioning. Equal update counts do not imply equal FLOPs; training time and parameter counts are recorded per candidate.", ""]
    (args.output / "benchmark_summary.md").write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "stop_rule_decision": result["stop_rule_decision"]}, indent=2))


if __name__ == "__main__":
    main()
