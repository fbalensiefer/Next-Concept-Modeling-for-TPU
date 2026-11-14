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
        return _models[params["engine_id"]]
    except KeyError as error:
        raise ValueError("unknown engine_id; call init_model first") from error


def dispatch(method, params):
    if method == "engine.ping":
        return {"status": "ready"}
    if method == "engine.describe":
        import flax
        import jax
        import jaxlib

        return {
            "engine": "ncm-jax-flax",
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "methods": [
                "engine.ping",
                "engine.describe",
                "init_model",
                "forward",
                "train_step",
                "save_checkpoint",
                "load_checkpoint",
            ],
        }
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
            "model": model,
            "optimizer": optimizer,
            "state": state,
            "sharding": sharding,
            "train_step": None,
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
            loss, (metrics, output) = entry["model"].apply(
                {"params": entry["state"].params},
                batch,
                entry["state"].codebooks,
                training=False,
                return_latent=False,
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
