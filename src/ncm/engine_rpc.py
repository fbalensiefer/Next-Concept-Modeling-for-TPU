"""JSON-RPC 2.0 stdio server for the JAX/Flax model engine."""

import json
import hashlib
import sys
import time
import uuid
from pathlib import Path

_models = {}


def robust_load_checkpoint(template, checkpoint_path):
    """Restore a complete compatible tree; reject missing or shape-mismatched leaves."""
    from flax import serialization
    import numpy as np

    restored = serialization.msgpack_restore(Path(checkpoint_path).read_bytes())
    return _strict_restore(template, restored, serialization, np)


def _strict_from_bytes(template, payload):
    from .checkpoint_audit import strict_from_bytes
    return strict_from_bytes(template, payload)


def _strict_restore(template, restored, serialization, np):
    from .checkpoint_audit import compare_trees

    expected = serialization.to_state_dict(template)
    actual = serialization.to_state_dict(restored)
    result = compare_trees(expected, actual)
    if result["status"] != "exact_match":
        details = ", ".join(f"{key}={value}" for key, value in result.items() if key != "status" and value)
        raise ValueError(f"checkpoint {result['status']}: {details}")
    return serialization.from_state_dict(template, actual)


def compute_reconstruction_metrics(target_tokens, decoded_tokens, rvq_indices_list,
                                   codebook_size=256, mask=None):
    """Compute batch reconstruction and per-stage RVQ utilization metrics."""
    import jax.numpy as jnp

    target = jnp.asarray(target_tokens)
    decoded = jnp.asarray(decoded_tokens)
    if target.shape != decoded.shape or target.ndim != 2:
        raise ValueError("reconstruction tokens must have matching [batch, length] shapes")
    valid = jnp.ones(target.shape, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    if valid.shape != target.shape:
        raise ValueError("reconstruction mask must match [batch, length]")
    valid_counts = valid.sum()
    ber = jnp.sum((target != decoded) & valid) / jnp.maximum(valid_counts, 1)
    exact = jnp.mean(jnp.all((target == decoded) | ~valid, axis=-1).astype(jnp.float32))
    perplexities = []
    assignment_counts = []
    for indices in rvq_indices_list:
        flat = jnp.asarray(indices).reshape(-1)
        flat = flat[flat >= 0]
        counts = jnp.bincount(flat, length=codebook_size)
        assignment_counts.append(counts)
        probabilities = counts / jnp.maximum(counts.sum(), 1)
        entropy = -jnp.sum(jnp.where(probabilities > 0,
                                     probabilities * jnp.log(probabilities), 0.0))
        perplexities.append(jnp.exp(entropy))
    return {"ber": ber, "exact_match": exact,
            "valid_bytes": valid_counts,
            "codebook_counts": jnp.stack(assignment_counts) if assignment_counts else jnp.zeros((0, codebook_size), jnp.int32),
            "codebook_perplexity": jnp.stack(perplexities) if perplexities else jnp.zeros((0,))}


def _codec_sha256(value):
    """Hash codec arrays in a stable order, independent of mapping order."""
    import numpy as np
    from flax import serialization
    from flax.traverse_util import flatten_dict

    # Normalize device arrays and pytree wrappers through the same msgpack
    # representation used by the checkpoint protocol before hashing.
    value = serialization.msgpack_restore(serialization.to_bytes(value))
    digest = hashlib.sha256()
    for path, leaf in sorted(flatten_dict(value).items()):
        array = np.asarray(leaf)
        digest.update("/".join(map(str, path)).encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class MethodNotFound(Exception):
    """Raised when a request names an unsupported engine operation."""


def _batch(data):
    import jax.numpy as jnp

    from ncm.training import Batch

    tokens = jnp.asarray(data["tokens"], jnp.int32)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, length]")
    shape = tokens.shape
    mask = jnp.asarray(
        data.get("mask", [[True] * shape[1] for _ in range(shape[0])]), jnp.bool_
    )
    fallback_bytes = jnp.asarray(
        data.get("fallback_bytes", [[0] * shape[1] for _ in range(shape[0])]),
        jnp.int32,
    )
    fallback_mask = jnp.asarray(
        data.get("fallback_mask", [[False] * shape[1] for _ in range(shape[0])]),
        jnp.bool_,
    )
    prompt_ids = data.get("prompt_ids")
    prompt_mask = data.get("prompt_mask")
    return Batch(
        tokens,
        mask,
        fallback_bytes,
        fallback_mask,
        None if prompt_ids is None else jnp.asarray(prompt_ids, jnp.int32),
        None if prompt_mask is None else jnp.asarray(prompt_mask, jnp.bool_),
    )


def _json_value(value):
    import jax
    import numpy as np

    value = jax.device_get(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite model result")
    return value


def _entry(params):
    try:
        entry = _models[params["engine_id"]]
    except KeyError as error:
        raise ValueError("unknown engine_id; call init_model first") from error
    if entry.get("kind") == "causal_ar":
        raise ValueError("engine_id is not an NCM codec model")
    return entry


def _ar_entry(params):
    try:
        entry = _models[params["engine_id"]]
    except KeyError as error:
        raise ValueError("unknown engine_id; call causal_ar_init first") from error
    if entry.get("kind") != "causal_ar":
        raise ValueError("engine_id is not a causal AR model")
    return entry


def _ar_batch(tokens, mask=None):
    import jax.numpy as jnp

    tokens = jnp.asarray(tokens, jnp.int32)
    if tokens.ndim != 2 or tokens.shape[1] < 1:
        raise ValueError("tokens must be nonempty [batch, length]")
    valid = jnp.ones(tokens.shape, jnp.bool_) if mask is None else jnp.asarray(mask, jnp.bool_)
    if valid.shape != tokens.shape:
        raise ValueError("mask shape must match tokens")
    return tokens, valid


def _verify_ar_proposals(model, model_params, prefixes, prefix_lengths, drafts,
                         proposal_probs, key, temperature, top_k):
    import jax
    import jax.numpy as jnp
    from ncm.autoregressive import shifted_inputs

    batch, max_prefix = prefixes.shape
    draft_len = drafts.shape[1]
    sequence = jnp.zeros((batch, max_prefix + draft_len + 1), jnp.int32)
    sequence = sequence.at[:, :max_prefix].set(prefixes)
    rows = jnp.arange(batch)[:, None]
    draft_positions = prefix_lengths[:, None] + jnp.arange(draft_len)[None, :]
    sequence = sequence.at[rows, draft_positions].set(drafts)
    valid_lengths = prefix_lengths + draft_len + 1
    valid = jnp.arange(sequence.shape[1])[None, :] < valid_lengths[:, None]
    logits = model.apply({"params": model_params}, shifted_inputs(sequence), valid)
    target_logits = jax.vmap(
        lambda row, start: jax.lax.dynamic_slice_in_dim(row, start, draft_len + 1, axis=0)
    )(logits, prefix_lengths)
    if temperature == 0:
        target_probs = jax.nn.one_hot(target_logits.argmax(-1), 256)
    else:
        target_logits = target_logits / temperature
        if top_k:
            cutoff = jax.lax.top_k(target_logits, top_k)[0][..., -1:]
            target_logits = jnp.where(target_logits >= cutoff, target_logits, -jnp.inf)
        target_probs = jax.nn.softmax(target_logits, -1)
    selected_p = jnp.take_along_axis(target_probs[:, :draft_len], drafts[..., None], -1)[..., 0]
    selected_q = jnp.take_along_axis(proposal_probs, drafts[..., None], -1)[..., 0]
    uniforms = jax.random.uniform(key, selected_p.shape)
    accepted = uniforms < jnp.minimum(1., selected_p / jnp.maximum(selected_q, 1e-30))
    prefix_accept = jnp.cumprod(accepted.astype(jnp.int32), axis=1).sum(1)
    rejection_index = jnp.minimum(prefix_accept, draft_len-1)
    row_ids = jnp.arange(batch)
    p_reject = target_probs[row_ids, rejection_index]
    q_reject = proposal_probs[row_ids, rejection_index]
    residual = jnp.maximum(p_reject-q_reject, 0.)
    residual = jnp.where(residual.sum(-1, keepdims=True) > 0, residual, p_reject)
    residual = residual / residual.sum(-1, keepdims=True)
    bonus_p = jnp.where((prefix_accept == draft_len)[:, None], target_probs[:, -1], residual)
    bonus = jax.random.categorical(jax.random.fold_in(key, 1),
        jnp.log(jnp.maximum(bonus_p, 1e-30))).astype(jnp.int32)
    accepted_mask = jnp.arange(draft_len)[None, :] < prefix_accept[:, None]
    returned = jnp.concatenate([jnp.where(accepted_mask, drafts, 0), bonus[:, None]], axis=1)
    return returned, prefix_accept, accepted_mask


def _verify_arrays(entry, prefixes, prefix_lengths, drafts, proposal_probs,
                   temperature, top_k, seed, trusted_proposals=False):
    """Verify JAX proposal arrays; trusted internal proposals avoid host copies."""
    import jax
    import jax.numpy as jnp
    import math
    import numpy as np

    if entry.get("kind") != "causal_ar":
        raise ValueError("engine_id is not a causal AR model")
    prefixes = jnp.asarray(prefixes, jnp.int32)
    drafts = jnp.asarray(drafts, jnp.int32)
    proposal_probs = jnp.asarray(proposal_probs, jnp.float32)
    if prefixes.ndim != 2 or drafts.ndim != 2 or proposal_probs.ndim != 3:
        raise ValueError("prefixes/drafts/proposal_probs have invalid rank")
    batch, prompt_len = prefixes.shape
    draft_len = drafts.shape[1]
    if drafts.shape[0] != batch or proposal_probs.shape != (batch, draft_len, 256):
        raise ValueError("proposal probabilities must be [batch,draft_len,256]")
    prefix_lengths = jnp.asarray(prefix_lengths, jnp.int32)
    if prefix_lengths.shape != (batch,) or bool(jnp.any(prefix_lengths < 0)) or bool(jnp.any(prefix_lengths > prompt_len)):
        raise ValueError("prefix_lengths must fit the padded prefix width")
    if int(prefix_lengths.max()) + draft_len + 1 > entry["model"].max_length or draft_len == 0:
        raise ValueError("prefix and draft exceed AR model capacity")
    if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
        raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")
    if not trusted_proposals:
        qsum = np.asarray(proposal_probs.sum(-1))
        prefix_values, draft_values = np.asarray(prefixes), np.asarray(drafts)
        valid_prefix_mask = np.arange(prompt_len)[None, :] < np.asarray(prefix_lengths)[:, None]
        if (np.any(prefix_values[valid_prefix_mask] < 0) or np.any(prefix_values[valid_prefix_mask] > 255)
                or np.any(draft_values < 0) or np.any(draft_values > 255)):
            raise ValueError("prefix and draft tokens must be byte IDs in 0..255")
        if np.any(~np.isfinite(qsum)) or np.any(np.abs(qsum-1.) > 1e-4) or np.any(np.asarray(proposal_probs) < 0):
            raise ValueError("proposal distributions must be finite, nonnegative and normalized")
    cache_key = (prompt_len, draft_len, batch, temperature, top_k)
    if cache_key not in entry["spec_verify"]:
        model = entry["model"]
        entry["spec_verify"][cache_key] = jax.jit(
            lambda model_params, pfx, pfx_lengths, drf, q, key: _verify_ar_proposals(
                model, model_params, pfx, pfx_lengths, drf, q, key, temperature, top_k
            )
        )
    returned, prefix_accept, accepted_mask = entry["spec_verify"][cache_key](
        entry["params"], prefixes, prefix_lengths, drafts, proposal_probs,
        jax.random.PRNGKey(seed & 0xffffffff),
    )
    return returned, prefix_accept, accepted_mask


def _derive_benchmark_seed(seed, prompt_len, batch, draft, repetition, round_index, stream):
    """Mirror rust/src/sweep.rs::derive_seed exactly for per-round RNG parity."""
    mask = (1 << 64) - 1

    def rotl(value, shift):
        value &= mask
        return ((value << shift) | (value >> (64 - shift))) & mask

    value = (
        (seed & mask)
        ^ ((prompt_len * 0x9e3779b97f4a7c15) & mask)
        ^ rotl(batch, 11)
        ^ rotl(draft, 23)
        ^ ((repetition * 0xd6e8feb86659fd93) & mask)
        ^ ((round_index * 0xa0761d6478bd642f) & mask)
        ^ (stream & mask)
    )
    value ^= value >> 30
    value = (value * 0xbf58476d1ce4e5b9) & mask
    value ^= value >> 27
    value = (value * 0x94d049bb133111eb) & mask
    value ^= value >> 31
    return value & 0xffffffff


def _pad_rows(rows, min_width=0):
    width = max(max((len(row) for row in rows), default=0), min_width)
    return ([list(row) + [0] * (width - len(row)) for row in rows],
            [len(row) for row in rows])


def _generate_and_verify_atomic(params):
    """Run one whole speculative observation, retaining q distributions in JAX."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        target = _models[params["target_engine_id"]]
        drafter = _models[params["drafter_engine_id"]]
    except KeyError as error:
        raise ValueError("unknown target or drafter engine_id") from error
    if target.get("kind") != "causal_ar":
        raise ValueError("target engine_id must identify a causal AR model")
    if drafter.get("kind") not in {"causal_ar", "ncm"}:
        raise ValueError("drafter engine_id must identify a supported model")

    prompts = [list(map(int, row)) for row in params["prompts"]]
    prompt_len = int(params["prompt_len"])
    output_len = int(params["output_len"])
    draft_len = int(params["draft_len"])
    temperature = float(params.get("temperature", 1.0))
    top_k = int(params.get("top_k", 0))
    seed = int(params["seed"])
    if not prompts or prompt_len < 0 or output_len < 1 or draft_len < 1:
        raise ValueError("atomic speculative request requires prompts and positive output/draft lengths")
    if any(len(row) != prompt_len for row in prompts):
        raise ValueError("each prompt must match prompt_len")
    if any(token < 0 or token > 255 for row in prompts for token in row):
        raise ValueError("prompt tokens must be byte IDs in 0..255")
    if not np.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
        raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")

    contexts = [row.copy() for row in prompts]
    batch_size = len(prompts)
    accepted_total = 0
    proposed_total = 0
    batch_rounds = 0
    sequence_rounds = 0
    prefix_sum = 0
    accepted_by_position = [0] * draft_len
    proposed_by_position = [0] * draft_len
    started = time.perf_counter()

    while any(len(row) - prompt_len < output_len for row in contexts):
        active = [i for i, row in enumerate(contexts)
                  if len(row) - prompt_len < output_len]
        remaining = min(output_len - (len(contexts[i]) - prompt_len) for i in active)
        width = min(draft_len, remaining)
        if width < 1:
            raise ValueError("speculative scheduler encountered an empty active row")
        batch_rounds += 1
        sequence_rounds += len(active)
        active_contexts = [contexts[i].copy() for i in active]
        prefix_sum += sum(len(row) for row in active_contexts)

        if drafter.get("kind") == "ncm":
            draft_prompts = (
                [[] for _ in active_contexts]
                if prompt_len == 0
                else [row[-prompt_len:] for row in active_contexts]
            )
            draft_prompt_array = jnp.asarray(draft_prompts, jnp.int32)
            draft_prompt_lengths = None
        else:
            draft_prompts, draft_prompt_lengths = _pad_rows(active_contexts, 1)
            draft_prompt_array = jnp.asarray(draft_prompts, jnp.int32)

        draft_seed = _derive_benchmark_seed(
            seed, prompt_len, batch_size, draft_len, 0, batch_rounds, 1
        )
        generated_tokens, proposal_probs = _generate_arrays(
            drafter, draft_prompt_array, draft_prompt_lengths, width,
            temperature, top_k, draft_seed
        )
        prefixes, prefix_lengths = _pad_rows(active_contexts, 0)
        verify_seed = _derive_benchmark_seed(
            seed, prompt_len, batch_size, draft_len, 0, batch_rounds, 2
        )
        verified_tokens, accepted_lengths, _ = _verify_arrays(
            target, jnp.asarray(prefixes, jnp.int32), prefix_lengths,
            generated_tokens, proposal_probs, temperature, top_k, verify_seed,
            trusted_proposals=True,
        )

        # Only compact token and count arrays cross from the accelerator to the
        # worker host. Proposal distributions never leave JAX memory.
        generated_host, verified_host, accepted_host = jax.device_get(
            (generated_tokens, verified_tokens, accepted_lengths)
        )
        generated_host = np.asarray(generated_host, dtype=np.int32).tolist()
        verified_host = np.asarray(verified_host, dtype=np.int32).tolist()
        accepted_host = np.asarray(accepted_host, dtype=np.int32).tolist()
        for local, global_index in enumerate(active):
            accepted = int(accepted_host[local])
            if accepted < 0 or accepted > width or len(generated_host[local]) != width:
                raise ValueError("verifier acceptance or proposal width is invalid")
            accepted_total += accepted
            proposed_total += width
            for position in range(width):
                proposed_by_position[position] += 1
                if position < accepted:
                    accepted_by_position[position] += 1
            contexts[global_index].extend(generated_host[local][:accepted])
            if len(contexts[global_index]) - prompt_len < output_len:
                bonus = verified_host[local][width]
                contexts[global_index].append(int(bonus))

    continuations = [
        [int(token) for token in row[prompt_len:prompt_len + output_len]]
        for row in contexts
    ]
    if any(len(row) != output_len for row in continuations):
        raise ValueError("atomic speculative output did not reach the requested length")
    return {
        "tokens": continuations,
        "accepted_tokens": accepted_total,
        "proposed_tokens": proposed_total,
        "accepted_by_position": accepted_by_position,
        "proposed_by_position": proposed_by_position,
        "batch_rounds": batch_rounds,
        "sequence_rounds": sequence_rounds,
        "prefix_sum": prefix_sum,
        "acceptance_rate": accepted_total / max(proposed_total, 1),
        "engine_elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def _generate_ncm(model, model_params, codebooks, prompts, length, key, temperature, top_k):
    import jax
    import jax.numpy as jnp
    from ncm.generator import SharedLevelGenerator
    from ncm.tokenizer import MultiscaleTokenizer

    batch, prompt_len = prompts.shape
    storage = model.rvq.max_length
    valid = jnp.broadcast_to(jnp.arange(storage) < length, (batch, storage))
    levels = jnp.stack([
        jnp.broadcast_to(jnp.arange(storage) < level_len, (batch, storage))
        for level_len in model.rvq.lengths
    ])
    targets = jnp.zeros(levels.shape, jnp.int32)
    latent_key, token_key = jax.random.split(key)
    generator = SharedLevelGenerator(model.rvq, model.transformer, model.config)
    prior_params = model_params["SharedLevelGenerator_0"]
    output = generator.apply(
        {"params": prior_params}, codebooks, levels, targets,
        teacher_forcing=False, source_mask=valid, temperature=1.,
        prompt_ids=prompts, use_cache=True, rngs={"sample": latent_key},
    )
    decoder = MultiscaleTokenizer(model.rvq, model.token_vocab_size)
    logits = decoder.apply(
        {"params": model_params["MultiscaleTokenizer_0"]}, output.representation,
        valid, method=decoder.decode,
    )
    if temperature == 0:
        tokens = logits.argmax(-1).astype(jnp.int32)
        probs = jax.nn.one_hot(tokens, model.token_vocab_size)
    else:
        logits = logits / temperature
        if top_k:
            cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
            logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
        probs = jax.nn.softmax(logits, -1)
        tokens = jax.random.categorical(token_key, logits).astype(jnp.int32)
    return tokens, probs


def _generate_ar_ragged(model, model_params, prompts, prompt_lengths, length, key,
                        temperature, top_k):
    """Cached AR generation for a batch whose prefixes have different lengths."""
    import jax
    import jax.numpy as jnp

    batch = prompts.shape[0]
    shape = (batch, model.max_length, model.config.heads,
             model.config.width // model.config.heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape))
                  for _ in range(model.config.layers))
    inputs = jnp.concatenate([jnp.full((batch, 1), 256, jnp.int32), prompts], axis=1)

    def prefill(cache, position):
        active = prompt_lengths > position
        _, cache = model.apply(
            {"params": model_params}, inputs[:, position],
            jnp.full((batch,), position, jnp.int32), cache, active,
            method=model.decode_step_ragged,
        )
        return cache, None

    cache, _ = jax.lax.scan(prefill, cache, jnp.arange(prompts.shape[1]))
    rows = jnp.arange(batch)
    last_index = jnp.maximum(prompt_lengths - 1, 0)
    previous = jnp.where(
        prompt_lengths > 0, prompts[rows, last_index], jnp.full((batch,), 256, jnp.int32)
    )

    def step(carry, index):
        previous, cache = carry
        positions = prompt_lengths + index
        logits, cache = model.apply(
            {"params": model_params}, previous, positions, cache,
            jnp.ones((batch,), jnp.bool_), method=model.decode_step_ragged,
        )
        if temperature == 0:
            tokens = logits.argmax(-1).astype(jnp.int32)
            probabilities = jax.nn.one_hot(tokens, 256)
        else:
            logits = logits / temperature
            if top_k:
                cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
                logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
            probabilities = jax.nn.softmax(logits, -1)
            tokens = jax.random.categorical(jax.random.fold_in(key, index), logits).astype(jnp.int32)
        return (tokens, cache), (tokens, probabilities)

    _, (tokens, probabilities) = jax.lax.scan(
        step, (previous, cache), jnp.arange(length)
    )
    return tokens.T, jnp.transpose(probabilities, (1, 0, 2))


def _generate_arrays(entry, prompts, prompt_lengths, max_new_tokens, temperature, top_k, seed):
    """Generate JAX arrays without converting intermediate proposals to JSON."""
    import jax
    import jax.numpy as jnp
    import math

    if prompts.ndim != 2 or prompts.shape[0] < 1:
        raise ValueError("prompts must have shape [nonempty batch, prompt_length]")
    if prompt_lengths is not None:
        prompt_lengths = jnp.asarray(prompt_lengths, jnp.int32)
        if prompt_lengths.shape != (prompts.shape[0],):
            raise ValueError("prompt_lengths must have one entry per prompt")
        if bool(jnp.any(prompt_lengths < 0)) or bool(jnp.any(prompt_lengths > prompts.shape[1])):
            raise ValueError("prompt_lengths must fit the padded prompt width")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
        raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")

    key = jax.random.PRNGKey(seed & 0xffffffff)
    if entry.get("kind") == "ncm":
        if max_new_tokens > entry["model"].rvq.max_length:
            raise ValueError("continuation exceeds NCM codec bucket")
        if prompts.shape[1] != entry["model"].config.prompt_len:
            raise ValueError("prompt length must match NCM model config")
        if prompt_lengths is not None and bool(jnp.any(prompt_lengths != prompts.shape[1])):
            raise ValueError("NCM prompts must use the configured fixed prompt length")
        if prompts.size and (int(prompts.min()) < 0 or int(prompts.max()) > 255):
            raise ValueError("prompt tokens must be byte IDs in 0..255")
        cache_key = (max_new_tokens, temperature, top_k)
        if cache_key not in entry["generate"]:
            model = entry["model"]
            entry["generate"][cache_key] = jax.jit(
                lambda model_params, codebooks, prompt, key: _generate_ncm(
                    model, model_params, codebooks, prompt, max_new_tokens, key,
                    temperature, top_k
                )
            )
        tokens, probabilities = entry["generate"][cache_key](
            entry["state"].params, entry["state"].codebooks, prompts, key
        )
        return tokens[:, :max_new_tokens], probabilities[:, :max_new_tokens]

    if entry.get("kind") != "causal_ar":
        raise ValueError("engine_id is not a causal AR model")
    max_prompt_len = prompts.shape[1] if prompt_lengths is None else int(prompt_lengths.max())
    if max_prompt_len + max_new_tokens > entry["model"].max_length:
        raise ValueError("prompt plus continuation exceeds max_length")
    if prompt_lengths is not None:
        cache_key = ("ragged", prompts.shape[1], max_new_tokens, temperature, top_k)
        if cache_key not in entry["generate"]:
            model = entry["model"]
            entry["generate"][cache_key] = jax.jit(
                lambda model_params, padded, lengths, rng: _generate_ar_ragged(
                    model, model_params, padded, lengths, max_new_tokens, rng,
                    temperature, top_k
                )
            )
        return entry["generate"][cache_key](
            entry["params"], prompts, prompt_lengths, key
        )

    cache_key = (max_new_tokens, temperature, top_k)
    if cache_key not in entry["generate"]:
        model = entry["model"]

        def generate(model_params, prompt, rng):
            batch_size, prompt_len = prompt.shape
            shape = (batch_size, model.max_length, model.config.heads,
                     model.config.width // model.config.heads)
            cache = tuple((jnp.zeros(shape), jnp.zeros(shape))
                          for _ in range(model.config.layers))
            inputs = jnp.concatenate([jnp.full((batch_size, 1), 256, jnp.int32), prompt], axis=1)

            def prefill(cache, i):
                _, cache = model.apply({"params": model_params}, inputs[:, i], i, cache,
                                       method=model.decode_step)
                return cache, None

            cache, _ = jax.lax.scan(prefill, cache, jnp.arange(prompt_len))

            def step(carry, i):
                previous, cache = carry
                logits, cache = model.apply({"params": model_params}, previous, prompt_len+i,
                                            cache, method=model.decode_step)
                if temperature > 0:
                    logits = logits / temperature
                    if top_k:
                        cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
                        logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
                    probabilities = jax.nn.softmax(logits, -1)
                    token = jax.random.categorical(jax.random.fold_in(rng, i), logits)
                else:
                    token = logits.argmax(-1)
                    probabilities = jax.nn.one_hot(token, 256)
                token = token.astype(jnp.int32)
                return (token, cache), (token, probabilities)

            _, (generated, probabilities) = jax.lax.scan(
                step, (inputs[:, -1], cache), jnp.arange(max_new_tokens)
            )
            return generated.T, jnp.transpose(probabilities, (1, 0, 2))

        entry["generate"][cache_key] = jax.jit(generate)
    return entry["generate"][cache_key](entry["params"], prompts, key)


def _generate_ar_tokens(entry, prompts, max_new_tokens, temperature, top_k, seed):
    """AR generation for callers that need tokens but not proposal distributions."""
    import jax
    import jax.numpy as jnp
    import math

    if entry.get("kind") != "causal_ar":
        raise ValueError("generate_tokens is supported for causal AR models")
    prompts = jnp.asarray(prompts, jnp.int32)
    if prompts.ndim != 2 or prompts.shape[0] < 1:
        raise ValueError("prompts must have shape [nonempty batch, prompt_length]")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
        raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")
    if prompts.shape[1] + max_new_tokens > entry["model"].max_length:
        raise ValueError("prompt plus continuation exceeds max_length")

    cache_key = ("tokens_only", max_new_tokens, temperature, top_k)
    if cache_key not in entry["generate"]:
        model = entry["model"]

        def generate(model_params, prompt, rng):
            batch_size, prompt_len = prompt.shape
            shape = (batch_size, model.max_length, model.config.heads,
                     model.config.width // model.config.heads)
            cache = tuple((jnp.zeros(shape), jnp.zeros(shape))
                          for _ in range(model.config.layers))
            inputs = jnp.concatenate([jnp.full((batch_size, 1), 256, jnp.int32), prompt], axis=1)

            def prefill(cache, index):
                _, cache = model.apply({"params": model_params}, inputs[:, index], index, cache,
                                       method=model.decode_step)
                return cache, None

            cache, _ = jax.lax.scan(prefill, cache, jnp.arange(prompt_len))

            def step(carry, index):
                previous, cache = carry
                logits, cache = model.apply(
                    {"params": model_params}, previous, prompt_len + index,
                    cache, method=model.decode_step
                )
                if temperature > 0:
                    logits = logits / temperature
                    if top_k:
                        cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
                        logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
                    token = jax.random.categorical(jax.random.fold_in(rng, index), logits)
                else:
                    token = logits.argmax(-1)
                token = token.astype(jnp.int32)
                return (token, cache), token

            _, tokens = jax.lax.scan(
                step, (inputs[:, -1], cache), jnp.arange(max_new_tokens)
            )
            return tokens.T

        entry["generate"][cache_key] = jax.jit(generate)
    tokens = entry["generate"][cache_key](
        entry["params"], prompts, jax.random.PRNGKey(seed & 0xffffffff)
    )
    return {"tokens": _json_value(tokens)}


def _generate_sequence(params):
    """Deterministic cached AR generation with nucleus filtering and log scores."""
    import math
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        entry = _models[params["engine_id"]]
    except KeyError as error:
        raise ValueError("unknown engine_id") from error
    if entry.get("kind") != "causal_ar":
        raise ValueError("generate_sequence currently requires a causal AR model")
    prompts = np.asarray(params["prompts"], dtype=np.int32)
    max_length = int(params["max_length"])
    temperature = float(params["temperature"])
    top_k = int(params["top_k"])
    top_p = float(params["top_p"])
    seed = int(params["seed"])
    if prompts.ndim != 2 or prompts.shape[0] < 1 or prompts.shape[1] < 1:
        raise ValueError("prompts must be a nonempty [batch, prompt_length] array")
    if max_length < 1 or prompts.shape[1] + max_length > entry["model"].max_length:
        raise ValueError("prompt plus max_length must fit the initialized model")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 <= top_k <= 256 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_k must be in 0..=256 and top_p in (0,1]")
    if np.any(prompts < 0) or np.any(prompts > 255):
        raise ValueError("prompt tokens must be byte IDs in 0..255")

    model = entry["model"]
    batch, prompt_len = prompts.shape
    shape = (batch, model.max_length, model.config.heads,
             model.config.width // model.config.heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(model.config.layers))
    inputs = jnp.concatenate([jnp.full((batch, 1), 256, jnp.int32),
                              jnp.asarray(prompts)], axis=1)
    def prefill(cache, index):
        _, cache = model.apply({"params": entry["params"]}, inputs[:, index], index,
                               cache, method=model.decode_step)
        return cache, None
    cache, _ = jax.lax.scan(prefill, cache, jnp.arange(prompt_len))
    rng = jax.random.PRNGKey(seed)

    def step(carry, index):
        previous, cache = carry
        logits, cache = model.apply({"params": entry["params"]}, previous,
                                    prompt_len + index, cache, method=model.decode_step)
        logits = logits / temperature
        if top_k:
            cutoff = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0][..., -1:]
            logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        remove = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs >= top_p
        threshold = jnp.min(jnp.where(remove, sorted_logits, jnp.inf), axis=-1, keepdims=True)
        threshold = jnp.where(jnp.any(remove, axis=-1, keepdims=True), threshold, -jnp.inf)
        logits = jnp.where(logits < threshold, -jnp.inf, logits)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token = jax.random.categorical(jax.random.fold_in(rng, index), logits).astype(jnp.int32)
        selected_log_prob = jnp.take_along_axis(log_probs, token[:, None], axis=-1)[:, 0]
        return (token, cache), (token, selected_log_prob, -jnp.sum(jnp.where(
            jnp.isfinite(log_probs), jnp.exp(log_probs) * log_probs, 0.0), axis=-1))
    started = time.perf_counter()
    _, (tokens, log_probs, entropies) = jax.lax.scan(step, (inputs[:, -1], cache),
                                                       jnp.arange(max_length))
    jax.block_until_ready(tokens)
    elapsed = time.perf_counter() - started
    tokens = np.asarray(tokens.T, dtype=np.int32)
    log_probs = np.asarray(log_probs.T, dtype=np.float64)
    entropies = np.asarray(entropies.T, dtype=np.float64)
    return {"tokens": tokens.tolist(), "log_probabilities": log_probs.tolist(),
            "metrics": {"sequence_count": int(batch), "token_count": int(batch * max_length),
                        "mean_log_probability": float(log_probs.mean()),
                        "mean_entropy_nats": float(entropies.mean()),
                        "generation_seconds": elapsed}}


def _generate_rvq_prior(params):
    """Sample level-major RVQ codes and decode them with a strict Phase 1 codec."""
    import math
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import serialization

    from ncm.autoregressive import AutoregressiveCategorical
    from ncm.config import LossConfig, TransformerConfig, experiment_rvq_config
    from ncm.grids import grid_matrices
    from ncm.prior_training import PriorTrainState, flatten_rvq_codes
    from ncm.training import Batch, NCM, initialize
    from ncm.tokenizer import MultiscaleTokenizer

    cfg = params.get("model_config")
    if not isinstance(cfg, dict):
        raise ValueError("model_config must be an object")
    rvq_config = experiment_rvq_config(cfg)
    transformer_config = TransformerConfig(cfg["width"], cfg["heads"], cfg["layers"])
    model = NCM(rvq_config, transformer_config, 256,
                LossConfig(prior=0., entropy=cfg.get("entropy", .01)))
    optimizer = optax.adam(float(cfg["learning_rate"]))
    zeros = jnp.zeros((1, rvq_config.max_length), jnp.int32)
    batch = Batch(zeros, jnp.ones_like(zeros, bool), jnp.zeros_like(zeros),
                  jnp.zeros_like(zeros, bool))
    codec_template = initialize(model, optimizer,
                                jax.random.key(int(params.get("seed", 0))), batch,
                                initialize_codebooks=False)
    codec_state = robust_load_checkpoint(codec_template, params["codec_checkpoint"])

    prior = AutoregressiveCategorical(transformer_config,
                                      sum(rvq_config.lengths),
                                      rvq_config.codebook_size)
    prior_inputs = jnp.full((1, prior.max_length), prior.vocab_size, jnp.int32)
    prior_mask = jnp.ones_like(prior_inputs, bool)
    prior_params = prior.init(jax.random.key(int(params.get("seed", 0)) + 1),
                              prior_inputs, prior_mask)["params"]
    prior_optimizer = optax.adam(float(cfg["learning_rate"]))
    prior_template = PriorTrainState(
        prior_params, prior_optimizer.init(prior_params), jnp.asarray(0, jnp.int32))
    prior_state = robust_load_checkpoint(prior_template, params["prior_checkpoint"])

    prompt = np.asarray(params.get("prompt", []), dtype=np.int32)
    if prompt.ndim != 1 or np.any(prompt < 0) or np.any(prompt > 255):
        raise ValueError("prompt must contain byte IDs in 0..255")
    if len(prompt) > rvq_config.max_length:
        raise ValueError("prompt exceeds the codec byte window")
    requested = int(params["max_length"])
    temperature, top_k, top_p = (float(params["temperature"]), int(params["top_k"]),
                                 float(params["top_p"]))
    seed = int(params["seed"])
    if requested < 1 or len(prompt) + requested > rvq_config.max_length:
        raise ValueError("prompt plus max_length must fit the codec byte window")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 <= top_k <= 256 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_k must be in 0..=256 and top_p in (0,1]")

    prompt_bytes = jnp.zeros((1, rvq_config.max_length), jnp.int32)
    prompt_mask = jnp.zeros_like(prompt_bytes, bool)
    if len(prompt):
        prompt_bytes = prompt_bytes.at[0, :len(prompt)].set(jnp.asarray(prompt))
        prompt_mask = prompt_mask.at[0, :len(prompt)].set(True)
    tokenized = MultiscaleTokenizer(rvq_config, 256).apply(
        {"params": codec_state.params["MultiscaleTokenizer_0"]},
        prompt_bytes, prompt_mask, jnp.zeros_like(prompt_bytes),
        jnp.zeros_like(prompt_mask), codec_state.codebooks)
    conditioned, conditioned_mask = flatten_rvq_codes(
        tokenized.rvq.indices, tokenized.rvq.mask, rvq_config.lengths)
    forced_codes, forced_mask = conditioned[0], conditioned_mask[0]

    def sample(params_tree, forced, forced_valid, key):
        cache = tuple((jnp.zeros((1, prior.max_length, transformer_config.heads,
                                  transformer_config.width // transformer_config.heads), jnp.float32),
                       jnp.zeros((1, prior.max_length, transformer_config.heads,
                                  transformer_config.width // transformer_config.heads), jnp.float32))
                      for _ in range(transformer_config.layers))

        def step(carry, position):
            previous, cache_state = carry
            logits, cache_state = prior.apply(
                {"params": params_tree}, previous, position, cache_state,
                method=prior.decode_step)
            raw_log_probs = jax.nn.log_softmax(logits, axis=-1)
            filtered = logits / temperature
            order = jnp.argsort(filtered, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(filtered, order, axis=-1)
            if top_k:
                keep_k = jnp.arange(sorted_logits.shape[-1])[None, :] < min(top_k, prior.vocab_size)
                sorted_logits = jnp.where(keep_k, sorted_logits, -jnp.inf)
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            remove = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs >= top_p
            sorted_logits = jnp.where(remove, -jnp.inf, sorted_logits)
            filtered_logits = jnp.full_like(filtered, -jnp.inf).at[
                jnp.arange(filtered.shape[0])[:, None], order].set(sorted_logits)
            sampling_log_probs = jax.nn.log_softmax(filtered_logits, axis=-1)
            sampled = jax.random.categorical(jax.random.fold_in(key, position),
                                             filtered_logits, axis=-1).astype(jnp.int32)
            known = forced_valid[position]
            token = jnp.where(known, forced[position], sampled)
            chosen_log_prob = jnp.take_along_axis(raw_log_probs, token[:, None], axis=-1)[:, 0]
            entropy = -jnp.sum(jnp.where(jnp.isfinite(sampling_log_probs),
                                         jnp.exp(sampling_log_probs) * sampling_log_probs, 0.), axis=-1)
            return (token, cache_state), (token, chosen_log_prob, entropy)

        return jax.lax.scan(step, (jnp.full((1,), prior.vocab_size, jnp.int32), cache),
                            jnp.arange(prior.max_length))[1]

    prior_started = time.perf_counter()
    sampled_codes, code_log_probs, code_entropies = jax.jit(sample)(
        prior_state.params, forced_codes, forced_mask,
        jax.random.PRNGKey(seed & 0xffffffff))
    jax.block_until_ready(sampled_codes)
    prior_seconds = time.perf_counter() - prior_started
    sampled_codes = sampled_codes[:, 0]

    decode_started = time.perf_counter()
    _, up = grid_matrices(rvq_config)
    representation = jnp.zeros((1, rvq_config.max_length, rvq_config.dim), jnp.float32)
    offset = 0
    for level, length in enumerate(rvq_config.lengths):
        indices = sampled_codes[offset:offset + length]
        vectors = codec_state.codebooks.embeddings[level, indices][None, ...]
        representation = representation + jnp.einsum(
            "ln,bnd->bld", up[level, :, :length], vectors)
        offset += length
    decoded_logits = MultiscaleTokenizer(rvq_config, 256).apply(
        {"params": codec_state.params["MultiscaleTokenizer_0"]}, representation,
        jnp.ones((1, rvq_config.max_length), bool), method=MultiscaleTokenizer.decode)
    byte_count = min(rvq_config.max_length, len(prompt) + requested)
    decoded_bytes = jnp.argmax(decoded_logits, axis=-1)[0, :byte_count].astype(jnp.int32)
    if len(prompt):
        # The lossy codec supplies generated content; retain literal prompt bytes.
        decoded_bytes = decoded_bytes.at[:len(prompt)].set(jnp.asarray(prompt))
    jax.block_until_ready(decoded_bytes)
    decode_seconds = time.perf_counter() - decode_started
    log_probs = np.asarray(code_log_probs[:, 0], dtype=np.float64)
    entropies = np.asarray(code_entropies[:, 0], dtype=np.float64)
    if not np.isfinite(log_probs).all() or not np.isfinite(entropies).all():
        raise ValueError("generation produced non-finite code probabilities or entropy")
    return {
        "tokens": [np.asarray(decoded_bytes, dtype=np.int32).tolist()],
        "log_probabilities": [log_probs.tolist()],
        "metrics": {
            "sequence_count": 1,
            "token_count": int(len(sampled_codes)),
            "mean_log_probability": float(log_probs.mean()),
            "mean_entropy_nats": float(entropies.mean()),
            "generation_seconds": float(prior_seconds + decode_seconds),
            "prior_generation_seconds": float(prior_seconds),
            "codec_decode_seconds": float(decode_seconds),
            "prompt_bytes": int(len(prompt)),
            "generated_bytes": int(max(0, byte_count - len(prompt))),
        },
    }


def dispatch(method, params):
    if method == "engine.ping":
        return {"status": "ready"}
    if method == "generate_sequence":
        request = dict(params)
        config = request.pop("config", {})
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        request.update(config)
        return _generate_sequence(request)
    if method == "generate_rvq_prior":
        request = dict(params)
        config = request.pop("config", {})
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        request.update(config)
        return _generate_rvq_prior(request)
    if method == "engine.describe":
        import platform
        import flax
        import jax
        import jaxlib
        import numpy
        import optax

        return {
            "engine": "ncm-jax-flax",
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": numpy.__version__,
            "python": platform.python_version(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "methods": [
                "engine.ping",
                "engine.describe",
                "init_model",
                "forward",
                "evaluate_reconstruction",
                "train_step",
                "probe.train",
                "causal_ar_init",
                "causal_ar_train_step",
                "generate_rvq_prior",
                "generate",
                "generate_tokens",
                "generate_and_verify_atomic",
                "causal_ar_evaluate",
                "speculative_verify",
                "save_checkpoint",
                "load_checkpoint",
                "save_codec",
                "load_codec",
            ],
        }
    if method == "causal_ar_init":
        import jax
        import optax
        from ncm.baselines import CausalAR
        from ncm.autoregressive import shifted_inputs
        from ncm.config import TransformerConfig

        config = params["config"]
        model = CausalAR(
            TransformerConfig(**config["transformer"]), int(config["max_length"])
        )
        optimizer = optax.adam(float(config.get("learning_rate", 3e-4)))
        tokens, mask = _ar_batch(params["tokens"], params.get("mask"))
        if tokens.shape[1] > model.max_length:
            raise ValueError("initialization batch exceeds max_length")
        model_inputs = shifted_inputs(tokens)
        variables = model.init(jax.random.PRNGKey(int(params["seed"])), model_inputs, mask)
        engine_id = uuid.uuid4().hex
        _models[engine_id] = {
            "kind": "causal_ar", "model": model, "params": variables["params"],
            "optimizer": optimizer, "opt_state": optimizer.init(variables["params"]),
            "step": 0, "train_step": {}, "forward": None,
            "generate": {},
            "spec_verify": {},
        }
        return {"engine_id": engine_id, "step": 0, "max_length": model.max_length}
    if method == "causal_ar_train_step":
        import jax
        import jax.numpy as jnp
        import optax
        from ncm.autoregressive import shifted_inputs
        from ncm.losses import cross_entropy

        entry = _ar_entry(params)
        tokens, mask = _ar_batch(params["tokens"], params.get("mask"))
        continuation_start = int(params.get("continuation_start", 0))
        if tokens.shape[1] > entry["model"].max_length or not 0 <= continuation_start < tokens.shape[1]:
            raise ValueError("invalid training sequence length or continuation_start")
        teacher_id = params.get("teacher_engine_id")
        weight = float(params.get("distillation_weight", 0.0))
        if not 0. <= weight <= 1. or (weight > 0. and not teacher_id):
            raise ValueError("distillation_weight must be in [0,1] and positive weight requires teacher_engine_id")
        teacher = _ar_entry({"engine_id": teacher_id}) if teacher_id else None
        if teacher is not None and tokens.shape[1] > teacher["model"].max_length:
            raise ValueError("training sequence exceeds teacher model max_length")
        cache_key = teacher_id is not None
        if cache_key not in entry["train_step"]:
            model, optimizer = entry["model"], entry["optimizer"]
            if teacher is not None:
                teacher_model = teacher["model"]
                @jax.jit
                def train_step(model_params, opt_state, teacher_params, tokens, mask, start, distill_weight):
                    selected = mask & (jnp.arange(tokens.shape[1])[None] >= start)
                    def objective(p):
                        logits = model.apply({"params": p}, shifted_inputs(tokens), mask)
                        ce = cross_entropy(logits, tokens, selected)
                        teacher_logits = jax.lax.stop_gradient(teacher_model.apply(
                            {"params": teacher_params}, shifted_inputs(tokens), mask))
                        probabilities = jax.nn.softmax(teacher_logits, -1)
                        kl = jnp.sum(probabilities * (jax.nn.log_softmax(teacher_logits)
                                    - jax.nn.log_softmax(logits)), -1)
                        kl = jnp.where(selected, kl, 0.).sum()/jnp.maximum(selected.sum(), 1)
                        return (1.-distill_weight)*ce + distill_weight*kl
                    loss, grads = jax.value_and_grad(objective)(model_params)
                    updates, opt_state = optimizer.update(grads, opt_state, model_params)
                    return optax.apply_updates(model_params, updates), opt_state, loss, optax.global_norm(grads)
            else:
                @jax.jit
                def train_step(model_params, opt_state, tokens, mask, start):
                    selected = mask & (jnp.arange(tokens.shape[1])[None] >= start)
                    def objective(p):
                        logits = model.apply({"params": p}, shifted_inputs(tokens), mask)
                        return cross_entropy(logits, tokens, selected)
                    loss, grads = jax.value_and_grad(objective)(model_params)
                    updates, opt_state = optimizer.update(grads, opt_state, model_params)
                    return optax.apply_updates(model_params, updates), opt_state, loss, optax.global_norm(grads)
            entry["train_step"][cache_key] = train_step
        started = time.perf_counter()
        if teacher is not None:
            model_params, opt_state, loss, grad_norm = entry["train_step"][cache_key](
                entry["params"], entry["opt_state"], teacher["params"], tokens, mask,
                jnp.asarray(continuation_start, jnp.int32), jnp.asarray(weight, jnp.float32))
        else:
            model_params, opt_state, loss, grad_norm = entry["train_step"][cache_key](
                entry["params"], entry["opt_state"], tokens, mask,
                jnp.asarray(continuation_start, jnp.int32))
        jax.block_until_ready(loss)
        entry["params"], entry["opt_state"] = model_params, opt_state
        entry["step"] += 1
        return {"step": entry["step"], "loss": _json_value(loss),
                "grad_norm": _json_value(grad_norm), "step_time_ms": (time.perf_counter()-started)*1000}
    if method == "generate_and_verify_atomic":
        return _generate_and_verify_atomic(params)
    if method == "generate_tokens":
        try:
            entry = _models[params["engine_id"]]
        except KeyError as error:
            raise ValueError("unknown engine_id; initialize a model first") from error
        return _generate_ar_tokens(
            entry,
            params["prompts"],
            int(params["max_new_tokens"]),
            float(params.get("temperature", 1.0)),
            int(params.get("top_k", 0)),
            int(params.get("seed", 0)),
        )
    if method == "generate":
        import jax
        import jax.numpy as jnp
        import math

        try:
            entry = _models[params["engine_id"]]
        except KeyError as error:
            raise ValueError("unknown engine_id; initialize a model first") from error
        prompts = jnp.asarray(params["prompts"], jnp.int32)
        if prompts.ndim != 2 or prompts.shape[0] < 1:
            raise ValueError("prompts must have shape [nonempty batch, prompt_length]")
        prompt_lengths = params.get("prompt_lengths")
        if prompt_lengths is not None:
            prompt_lengths = jnp.asarray(prompt_lengths, jnp.int32)
            if prompt_lengths.shape != (prompts.shape[0],):
                raise ValueError("prompt_lengths must have one entry per prompt")
            if bool(jnp.any(prompt_lengths < 0)) or bool(jnp.any(prompt_lengths > prompts.shape[1])):
                raise ValueError("prompt_lengths must fit the padded prompt width")
        max_new_tokens = int(params["max_new_tokens"])
        temperature = float(params.get("temperature", 1.0))
        top_k = int(params.get("top_k", 0))
        seed = int(params.get("seed", 0))
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
            raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")
        if entry.get("kind") == "ncm":
            if max_new_tokens > entry["model"].rvq.max_length:
                raise ValueError("continuation exceeds NCM codec bucket")
            if prompts.shape[1] != entry["model"].config.prompt_len:
                raise ValueError("prompt length must match NCM model config")
            if prompt_lengths is not None and bool(jnp.any(prompt_lengths != prompts.shape[1])):
                raise ValueError("NCM prompts must use the configured fixed prompt length")
            if prompts.size and (int(prompts.min()) < 0 or int(prompts.max()) > 255):
                raise ValueError("prompt tokens must be byte IDs in 0..255")
            cache_key = (max_new_tokens, temperature, top_k)
            if cache_key not in entry["generate"]:
                model = entry["model"]
                entry["generate"][cache_key] = jax.jit(
                    lambda model_params, codebooks, prompt, key: _generate_ncm(
                        model, model_params, codebooks, prompt, max_new_tokens, key,
                        temperature, top_k
                    )
                )
            tokens, probabilities = entry["generate"][cache_key](
                entry["state"].params, entry["state"].codebooks, prompts,
                jax.random.PRNGKey(seed & 0xffffffff)
            )
            # The decoder emits the full codec bucket; RPC callers request a
            # concrete continuation length and must receive exactly that many
            # tokens and proposal distributions.
            tokens = tokens[:, :max_new_tokens]
            probabilities = probabilities[:, :max_new_tokens]
            return {"tokens": _json_value(tokens), "probabilities": _json_value(probabilities)}
        entry = _ar_entry(params)
        max_prompt_len = prompts.shape[1] if prompt_lengths is None else int(prompt_lengths.max())
        if max_prompt_len + max_new_tokens > entry["model"].max_length:
            raise ValueError("prompt plus continuation exceeds max_length")
        if prompt_lengths is not None:
            cache_key = ("ragged", prompts.shape[1], max_new_tokens, temperature, top_k)
            if cache_key not in entry["generate"]:
                model = entry["model"]
                entry["generate"][cache_key] = jax.jit(
                    lambda model_params, padded, lengths, key: _generate_ar_ragged(
                        model, model_params, padded, lengths, max_new_tokens, key,
                        temperature, top_k
                    )
                )
            tokens, probabilities = entry["generate"][cache_key](
                entry["params"], prompts, prompt_lengths,
                jax.random.PRNGKey(seed & 0xffffffff)
            )
            return {"tokens": _json_value(tokens), "probabilities": _json_value(probabilities)}
        cache_key = (max_new_tokens, temperature, top_k)
        if cache_key not in entry["generate"]:
            model = entry["model"]
            def generate(model_params, prompt, key):
                batch_size, prompt_len = prompt.shape
                shape = (batch_size, model.max_length, model.config.heads,
                         model.config.width // model.config.heads)
                cache = tuple((jnp.zeros(shape), jnp.zeros(shape))
                              for _ in range(model.config.layers))
                inputs = jnp.concatenate([jnp.full((batch_size, 1), 256, jnp.int32), prompt], axis=1)
                def prefill(cache, i):
                    _, cache = model.apply({"params": model_params}, inputs[:, i], i, cache,
                                           method=model.decode_step)
                    return cache, None
                cache, _ = jax.lax.scan(prefill, cache, jnp.arange(prompt_len))
                def step(carry, i):
                    previous, cache = carry
                    logits, cache = model.apply({"params": model_params}, previous, prompt_len+i,
                                                cache, method=model.decode_step)
                    if temperature > 0:
                        logits = logits / temperature
                        if top_k:
                            cutoff = jax.lax.top_k(logits, top_k)[0][..., -1:]
                            logits = jnp.where(logits >= cutoff, logits, -jnp.inf)
                        probabilities = jax.nn.softmax(logits, -1)
                        token = jax.random.categorical(jax.random.fold_in(key, i), logits)
                    else:
                        token = logits.argmax(-1)
                        probabilities = jax.nn.one_hot(token, 256)
                    token = token.astype(jnp.int32)
                    return (token, cache), (token, probabilities)
                _, (generated, probabilities) = jax.lax.scan(
                    step, (inputs[:, -1], cache), jnp.arange(max_new_tokens)
                )
                return generated.T, jnp.transpose(probabilities, (1, 0, 2))
            entry["generate"][cache_key] = jax.jit(generate)
        generated = entry["generate"][cache_key](
            entry["params"], prompts, jax.random.PRNGKey(seed & 0xffffffff)
        )
        tokens, probabilities = generated
        return {"tokens": _json_value(tokens), "probabilities": _json_value(probabilities)}
    if method == "causal_ar_evaluate":
        import jax
        import jax.numpy as jnp
        import optax
        from ncm.autoregressive import shifted_inputs

        entry = _ar_entry(params)
        tokens, mask = _ar_batch(params["tokens"], params.get("mask"))
        prompt_len = int(params.get("prompt_len", 0))
        if tokens.shape[1] > entry["model"].max_length or not 0 <= prompt_len < tokens.shape[1]:
            raise ValueError("invalid evaluation sequence length or prompt_len")
        if entry["forward"] is None:
            model = entry["model"]
            entry["forward"] = jax.jit(
                lambda model_params, x, valid: model.apply(
                    {"params": model_params}, shifted_inputs(x), valid
                )
            )
        logits = entry["forward"](entry["params"], tokens, mask)
        token_nll = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
        selected = mask & (jnp.arange(tokens.shape[1])[None] >= prompt_len)
        token_ll = jnp.where(selected, -token_nll, 0.)
        count = int(selected.sum())
        if count == 0:
            raise ValueError("evaluation split contains no continuation tokens")
        total_nll = jnp.where(selected, token_nll, 0.).sum()
        mean_nll = total_nll / count
        jax.block_until_ready(mean_nll)
        return {"nll": _json_value(total_nll), "token_count": count,
                "mean_nll": _json_value(mean_nll), "perplexity": _json_value(jnp.exp(mean_nll)),
                "log_likelihoods": _json_value(token_ll)}
    if method == "speculative_verify":
        import jax
        import jax.numpy as jnp
        import math
        import numpy as np
        from ncm.autoregressive import shifted_inputs

        entry = _ar_entry(params)
        prefixes = jnp.asarray(params["prefixes"], jnp.int32)
        drafts = jnp.asarray(params["draft_tokens"], jnp.int32)
        proposal_probs = jnp.asarray(params["proposal_probs"], jnp.float32)
        temperature = float(params.get("temperature", 1.0))
        top_k = int(params.get("top_k", 0))
        if prefixes.ndim != 2 or drafts.ndim != 2 or proposal_probs.ndim != 3:
            raise ValueError("prefixes/drafts/proposal_probs have invalid rank")
        batch, prompt_len = prefixes.shape
        draft_len = drafts.shape[1]
        if drafts.shape[0] != batch or proposal_probs.shape != (batch, draft_len, 256):
            raise ValueError("proposal probabilities must be [batch,draft_len,256]")
        prefix_lengths = params.get("prefix_lengths")
        if prefix_lengths is None:
            prefix_lengths = jnp.full((batch,), prompt_len, jnp.int32)
        else:
            prefix_lengths = jnp.asarray(prefix_lengths, jnp.int32)
        if prefix_lengths.shape != (batch,) or bool(jnp.any(prefix_lengths < 0)) or bool(jnp.any(prefix_lengths > prompt_len)):
            raise ValueError("prefix_lengths must fit the padded prefix width")
        if int(prefix_lengths.max()) + draft_len + 1 > entry["model"].max_length or draft_len == 0:
            raise ValueError("prefix and draft exceed AR model capacity")
        if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
            raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")
        qsum = np.asarray(proposal_probs.sum(-1))
        prefix_values, draft_values = np.asarray(prefixes), np.asarray(drafts)
        valid_prefix_mask = np.arange(prompt_len)[None, :] < np.asarray(prefix_lengths)[:, None]
        if (np.any(prefix_values[valid_prefix_mask] < 0) or np.any(prefix_values[valid_prefix_mask] > 255)
                or np.any(draft_values < 0) or np.any(draft_values > 255)):
            raise ValueError("prefix and draft tokens must be byte IDs in 0..255")
        if np.any(~np.isfinite(qsum)) or np.any(np.abs(qsum-1.) > 1e-4) or np.any(np.asarray(proposal_probs) < 0):
            raise ValueError("proposal distributions must be finite, nonnegative and normalized")
        cache_key = (prompt_len, draft_len, batch, temperature, top_k)
        if cache_key not in entry["spec_verify"]:
            model = entry["model"]
            entry["spec_verify"][cache_key] = jax.jit(
                lambda model_params, pfx, pfx_lengths, drf, q, key: _verify_ar_proposals(
                    model, model_params, pfx, pfx_lengths, drf, q, key, temperature, top_k
                )
            )
        returned, prefix_accept, accepted_mask = entry["spec_verify"][cache_key](
            entry["params"], prefixes, prefix_lengths, drafts, proposal_probs,
            jax.random.PRNGKey(int(params.get("seed", 0)) & 0xffffffff),
        )
        jax.block_until_ready(returned)
        return {"tokens": _json_value(returned), "accepted_lengths": _json_value(prefix_accept),
                "acceptance_rate": _json_value(prefix_accept.sum()/max(batch*draft_len, 1)),
                "accepted_mask": _json_value(accepted_mask)}
    if method == "probe.train":
        import numpy as np

        from ncm.probe import train_probe

        train = np.asarray(params["train"], dtype=np.int32)
        evaluation = np.asarray(params["evaluation"], dtype=np.int32)
        nll, _ = train_probe(
            train,
            evaluation,
            tuple(int(x) for x in params["lengths"]),
            int(params["level"]),
            int(params["seed"]),
            steps=int(params.get("steps", 200)),
            batch_size=int(params.get("batch_size", 64)),
            causal=bool(params["causal"]),
        )
        return {"nll": nll}
    if method == "init_model":
        import jax
        import optax

        from ncm import (
            CoarseConfig,
            LossConfig,
            NCM,
            NCMConfig,
            RVQConfig,
            TransformerConfig,
            initialize,
        )

        config = params["config"]
        rvq_config = dict(config["rvq"])
        rvq_config["lengths"] = tuple(rvq_config["lengths"])
        rvq_config["coarse"] = CoarseConfig(**rvq_config.get("coarse", {}))
        model = NCM(
            RVQConfig(**rvq_config),
            TransformerConfig(**config.get("transformer", {})),
            int(config.get("token_vocab_size", 256)),
            LossConfig(**config.get("losses", {})),
            NCMConfig(**config.get("model", {})),
        )
        optimizer = optax.adam(float(config.get("learning_rate", 3e-4)))
        batch_size = int(config.get("init_batch_size", 1))
        batch = _batch(
            {"tokens": [[0] * model.rvq.max_length for _ in range(batch_size)]}
        )

        device_config = config.get("device", {})
        platform = device_config.get("platform")
        devices = jax.devices(platform) if platform else jax.devices()
        device_index = int(device_config.get("index", 0))
        if device_index < 0 or device_index >= len(devices):
            raise ValueError("device index is outside the available device set")
        device = devices[device_index]
        sharding = jax.sharding.SingleDeviceSharding(device)
        batch = jax.device_put(batch, sharding)
        with jax.default_device(device):
            state = initialize(
                model,
                optimizer,
                jax.random.PRNGKey(int(params["seed"])),
                batch,
            )

        engine_id = uuid.uuid4().hex
        _models[engine_id] = {
            "kind": "ncm",
            "model": model,
            "optimizer": optimizer,
            "state": state,
            "sharding": sharding,
            "train_step": {},
            "forward": None,
            "generate": {},
        }
        return {
            "engine_id": engine_id,
            "step": int(state.step),
            "backend": device.platform,
            "devices": [str(device)],
            "sharding": str(sharding),
        }

    if method in ("forward", "evaluate_reconstruction", "train_step", "save_checkpoint", "load_checkpoint",
                  "save_codec", "load_codec"):
        import jax
        import jax.numpy as jnp
        from flax import serialization

        stored = _models.get(params.get("engine_id"))
        ar_checkpoint = (
            method in ("save_checkpoint", "load_checkpoint")
            and stored is not None
            and stored.get("kind") == "causal_ar"
        )
        entry = None if ar_checkpoint else _entry(params)
        if method in ("forward", "evaluate_reconstruction"):
            batch = jax.device_put(_batch(params["batch"]), entry["sharding"])
            if entry["forward"] is None:
                model = entry["model"]
                entry["forward"] = jax.jit(
                    lambda model_params, codebooks, batch: model.apply(
                        {"params": model_params},
                        batch,
                        codebooks,
                        training=False,
                        return_latent=False,
                    )
                )
            loss, (metrics, output) = entry["forward"](
                entry["state"].params, entry["state"].codebooks, batch
            )
            if method == "evaluate_reconstruction":
                from ncm.tokenizer import MultiscaleTokenizer
                tokenizer = MultiscaleTokenizer(entry["model"].rvq,
                                                entry["model"].token_vocab_size)
                reconstruction = tokenizer.apply(
                    {"params": entry["state"].params["MultiscaleTokenizer_0"]},
                    batch.tokens, batch.mask, batch.fallback_bytes,
                    batch.fallback_mask, entry["state"].codebooks,
                    training=False,
                )
                decoded = jnp.argmax(reconstruction.logits, axis=-1).astype(batch.tokens.dtype)
                metrics = compute_reconstruction_metrics(
                    batch.tokens, decoded, reconstruction.rvq.indices,
                    codebook_size=entry["model"].rvq.codebook_size,
                    mask=batch.mask,
                )
                return {key: _json_value(value) for key, value in metrics.items()}
            return {
                "loss": _json_value(loss),
                "metrics": _json_value(metrics),
                "codes": _json_value(output.indices),
            }

        if method == "train_step":
            from ncm.training import make_prior_train_step, make_train_step

            prior_only = bool(params.get("prior_only", False))
            sequence_distillation = bool(params.get("sequence_distillation", False))
            if prior_only and sequence_distillation:
                raise ValueError("prior_only and sequence_distillation are mutually exclusive")
            mode = "prior" if prior_only else "sequence" if sequence_distillation else "joint"

            if mode not in entry["train_step"]:
                step_fn = (make_train_step(entry["model"], entry["optimizer"])
                           if mode == "joint" else
                           make_prior_train_step(entry["model"], entry["optimizer"],
                                                 sequence_distillation=mode == "sequence"))
                entry["train_step"][mode] = jax.jit(step_fn)
            started = time.perf_counter()
            state, metrics = entry["train_step"][mode](
                entry["state"],
                jax.device_put(_batch(params["batch"]), entry["sharding"]),
                jnp.asarray(int(params.get("phase", 0))),
                jax.random.PRNGKey(int(params["seed"])),
            )
            jax.block_until_ready(metrics["loss"])
            entry["state"] = state
            values = _json_value(metrics)
            return {
                "step": int(state.step),
                "loss": values["loss"],
                "grad_norm": values["grad_norm"],
                "step_time_ms": (time.perf_counter() - started) * 1000,
                "metrics": values,
            }

        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("path must be nonempty")
        from pathlib import Path

        if method in ("save_codec", "load_codec"):
            if entry.get("kind") != "ncm":
                raise ValueError("codec checkpoints require an NCM engine")
            codec_path = Path(path)
            if method == "save_codec":
                codec_path.parent.mkdir(parents=True, exist_ok=True)
                value = {"tokenizer_params": entry["state"].params["MultiscaleTokenizer_0"],
                         "codebooks": entry["state"].codebooks}
                codec_path.write_bytes(serialization.to_bytes(value))
                return {"path": str(codec_path), "step": int(entry["state"].step),
                        "codec_sha256": _codec_sha256(value)}
            template = {"tokenizer_params": entry["state"].params["MultiscaleTokenizer_0"],
                        "codebooks": entry["state"].codebooks}
            restored = serialization.msgpack_restore(codec_path.read_bytes())
            if isinstance(restored, dict) and set(restored) == {"tokenizer_params", "codebooks"}:
                value = robust_load_checkpoint(template, codec_path)
            elif isinstance(restored, dict) and isinstance(restored.get("params"), dict):
                params = restored["params"]
                if "MultiscaleTokenizer_0" not in params or "codebooks" not in restored:
                    raise ValueError("full checkpoint is missing tokenizer parameters or codebooks")
                value = {"tokenizer_params": params["MultiscaleTokenizer_0"],
                         "codebooks": restored["codebooks"]}
                # Validate the extracted full-state leaves against the initialized model.
                value = _strict_restore(template, serialization.to_state_dict(value),
                                        serialization, __import__("numpy"))
            else:
                raise ValueError("unsupported codec checkpoint schema")
            parameters = dict(entry["state"].params)
            parameters["MultiscaleTokenizer_0"] = value["tokenizer_params"]
            entry["state"] = entry["state"].replace(
                params=parameters, codebooks=value["codebooks"], step=0)
            entry["state"] = entry["state"].replace(
                opt_state=entry["optimizer"].init(entry["state"].params))
            return {"path": str(codec_path), "step": int(entry["state"].step),
                    "codec_sha256": _codec_sha256(value)}

        if method == "save_checkpoint":
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if ar_checkpoint:
                ar = _ar_entry(params)
                state = {
                    "params": ar["params"],
                    "opt_state": ar["opt_state"],
                    "step": ar["step"],
                }
                output_path.write_bytes(serialization.to_bytes(state))
                return {"path": str(output_path), "step": int(ar["step"])}
            output_path.write_bytes(serialization.to_bytes(entry["state"]))
            return {"path": str(output_path), "step": int(entry["state"].step)}

        if ar_checkpoint:
            ar = _ar_entry(params)
            template = {
                "params": ar["params"],
                "opt_state": ar["opt_state"],
                "step": ar["step"],
            }
            state = _strict_from_bytes(template, Path(path).read_bytes())
            ar["params"], ar["opt_state"], ar["step"] = (
                state["params"], state["opt_state"], int(state["step"])
            )
            return {"path": path, "step": ar["step"]}
        entry["state"] = _strict_from_bytes(entry["state"], Path(path).read_bytes())
        return {"path": path, "step": int(entry["state"].step)}

    raise MethodNotFound(method)


def response(request):
    request_id = request.get("id") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32600, "message": "invalid JSON-RPC 2.0 request"},
            }
        if isinstance(request_id, bool) or not isinstance(
            request_id, (str, int, float)
        ):
            raise ValueError("request id must be string or number")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str) or not method or not isinstance(params, dict):
            raise ValueError("method must be nonempty and params an object")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": dispatch(method, params),
        }
    except MethodNotFound as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": str(error)},
        }
    except Exception as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(error)},
        }


def serve(source=None, sink=None):
    source = sys.stdin if source is None else source
    sink = sys.stdout if sink is None else sink
    for line in source:
        try:
            request = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            result = response(request)
        except (json.JSONDecodeError, ValueError) as error:
            result = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(error)},
            }
        if "error" in result:
            print(
                f"JSON-RPC error {result['error']['code']}: "
                f"{result['error']['message']}",
                file=sys.stderr,
                flush=True,
            )
        sink.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        sink.flush()


if __name__ == "__main__":
    serve()
