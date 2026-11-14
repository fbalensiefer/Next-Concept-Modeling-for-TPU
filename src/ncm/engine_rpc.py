"""Strict JSON-RPC 2.0 stdio endpoint for the JAX/Flax model engine."""
import json
import sys


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
            "methods": ["engine.ping", "engine.describe"],
        }
    raise KeyError(f"unknown method: {method}")


def response(request):
    request_id = request.get("id")
    try:
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            raise ValueError("invalid JSON-RPC 2.0 request")
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int, float)):
            raise ValueError("request id must be a string or number")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str) or not method or not (isinstance(params, (dict, list)) or params is None):
            raise ValueError("invalid method or params")
        result = dispatch(method, params)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except KeyError as error:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": str(error)}}
    except Exception as error:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": str(error)}}


def serve(source=None, sink=None):
    source = sys.stdin if source is None else source
    sink = sys.stdout if sink is None else sink
    for line in source:
        try:
            request = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")))
            result = response(request)
        except json.JSONDecodeError as error:
            result = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}}
        except ValueError as error:
            result = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}}
        sink.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        sink.flush()


if __name__ == "__main__":
    serve()
