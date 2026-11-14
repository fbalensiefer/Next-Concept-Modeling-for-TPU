"""JSON-RPC 2.0 stdio server for the JAX/Flax model engine."""

import json
import sys
import time
import uuid

_models = {}


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


def _verify_ar_proposals(model, model_params, prefixes, drafts, proposal_probs, key,
                         temperature, top_k):
    import jax
    import jax.numpy as jnp
    from ncm.autoregressive import shifted_inputs

    batch, prompt_len = prefixes.shape
    draft_len = drafts.shape[1]
    sequence = jnp.concatenate([prefixes, drafts], axis=1)
    # One trailing placeholder lets the same causal pass produce the bonus
    # distribution conditioned on the full accepted draft. Its value is masked
    # from every distribution because causal attention only sees earlier tokens.
    sequence = jnp.concatenate([sequence, jnp.zeros((batch, 1), jnp.int32)], axis=1)
    logits = model.apply({"params": model_params}, shifted_inputs(sequence),
                         jnp.ones_like(sequence, jnp.bool_))
    target_logits = logits[:, prompt_len:prompt_len+draft_len+1]
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


def dispatch(method, params):
    if method == "engine.ping":
        return {"status": "ready"}
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
                "train_step",
                "probe.train",
                "causal_ar_init",
                "causal_ar_train_step",
                "generate",
                "causal_ar_evaluate",
                "speculative_verify",
                "save_checkpoint",
                "load_checkpoint",
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
            return {"tokens": _json_value(tokens), "probabilities": _json_value(probabilities)}
        entry = _ar_entry(params)
        if prompts.shape[1] < 1:
            raise ValueError("causal AR prompts must be nonempty")
        if prompts.shape[1] + max_new_tokens > entry["model"].max_length:
            raise ValueError("prompt plus continuation exceeds max_length")
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
        if prompt_len + draft_len + 1 > entry["model"].max_length or prompt_len == 0 or draft_len == 0:
            raise ValueError("prefix and draft exceed AR model capacity")
        if not math.isfinite(temperature) or temperature < 0 or not 0 <= top_k <= 256:
            raise ValueError("temperature must be finite and nonnegative; top_k must be in 0..=256")
        qsum = np.asarray(proposal_probs.sum(-1))
        prefix_values, draft_values = np.asarray(prefixes), np.asarray(drafts)
        if (np.any(prefix_values < 0) or np.any(prefix_values > 255)
                or np.any(draft_values < 0) or np.any(draft_values > 255)):
            raise ValueError("prefix and draft tokens must be byte IDs in 0..255")
        if np.any(~np.isfinite(qsum)) or np.any(np.abs(qsum-1.) > 1e-4) or np.any(np.asarray(proposal_probs) < 0):
            raise ValueError("proposal distributions must be finite, nonnegative and normalized")
        cache_key = (prompt_len, draft_len, batch, temperature, top_k)
        if cache_key not in entry["spec_verify"]:
            model = entry["model"]
            entry["spec_verify"][cache_key] = jax.jit(
                lambda model_params, pfx, drf, q, key: _verify_ar_proposals(
                    model, model_params, pfx, drf, q, key, temperature, top_k
                )
            )
        returned, prefix_accept, accepted_mask = entry["spec_verify"][cache_key](
            entry["params"], prefixes, drafts, proposal_probs,
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
            "train_step": None,
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

    if method in ("forward", "train_step", "save_checkpoint", "load_checkpoint"):
        import jax
        import jax.numpy as jnp
        from flax import serialization

        entry = _entry(params)
        if method == "forward":
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
            return {
                "loss": _json_value(loss),
                "metrics": _json_value(metrics),
                "codes": _json_value(output.indices),
            }

        if method == "train_step":
            from ncm.training import make_train_step

            if entry["train_step"] is None:
                entry["train_step"] = jax.jit(
                    make_train_step(entry["model"], entry["optimizer"])
                )
            started = time.perf_counter()
            state, metrics = entry["train_step"](
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
        if method == "save_checkpoint":
            from pathlib import Path

            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(serialization.to_bytes(entry["state"]))
            return {"path": str(output_path), "step": int(entry["state"].step)}

        from pathlib import Path

        entry["state"] = serialization.from_bytes(
            entry["state"], Path(path).read_bytes()
        )
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
