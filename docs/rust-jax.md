# Rust orchestration and the JAX engine

Rust owns the supported CLI, data preparation, experiment scheduling, statistics, FLOP estimates, validation, and report generation. Python is limited to JAX/Flax model definitions and tensor execution in `src/ncm/engine_rpc.py`. `ncm-run` has no Python workload-launcher path.

## Setup and commands

Install Rust and Python 3.12 with JAX, Flax, Optax, NumPy, and pytest:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export NCM_PYTHON="$PWD/.venv/bin/python"
cargo run --bin ncm-run -- --list
cargo run --bin ncm-run -- engine-info
```

`NCM_PYTHON` selects the interpreter for the engine only and defaults to `python3`. `NCM_RPC_TIMEOUT_MS` controls each request timeout (default 300000 ms). `NCM_TECTONIC` selects the paper compiler (default `tectonic`).

Train the codec model through Rust orchestration:

```sh
cargo run --bin ncm-run -- train \
  --config examples/text/config.json \
  --output runs/text-rust
```

Run the causal-AR evaluation and drafter comparison:

```sh
cargo run --bin ncm-run -- run-decisive-benchmark \
  --model-config examples/text/ar_config.json \
  --train examples/text/train.txt \
  --heldout examples/text/validation.txt \
  --steps 10 --batch-size 2 --prompt-len 1 --seed 7 \
  --output runs/decisive-ar

cargo run --bin ncm-run -- run-drafter-comparison \
  --target-config examples/text/ar_config.json \
  --drafter small-ar=examples/text/ar_config.json \
  --prompts examples/text/prompts.txt \
  --max-new-tokens 8 --draft-len 2 --seed 7 \
  --output runs/drafter-ar
```

These compact commands are smoke examples. The decisive runner evaluates a configured causal AR model; the drafter runner compares configured causal AR drafters with direct target generation. Their analytic FLOP figures use Rust estimates. They do not report compiler FLOPs or BPB without the corresponding measurements and evaluation data.

## JSON-RPC over stdio

`EngineClient::start` launches one long-lived worker process:

```text
NCM_PYTHON -u -m ncm.engine_rpc
```

Rust sends one JSON-RPC 2.0 object per line to stdin. The worker sends exactly one response with the matching `id` per line on stdout. Worker diagnostics go to inherited stderr; stdout is reserved for protocol messages. Requests are sequential. Tensor batches and masks are JSON arrays in typed structs declared in [`rpc.rs`](../rust/src/rpc.rs). There is no Python script dispatch and no embedded Python runtime.

Every request has this envelope:

```json
{"jsonrpc":"2.0","id":1,"method":"engine.describe","params":{}}
```

Responses contain either `result` or `error`:

```json
{"jsonrpc":"2.0","id":1,"result":{"methods":["engine.ping"]}}
```

Errors propagate to Rust. The client checks process liveness before each call, validates response version and ID, and reports invalid responses, RPC errors, panics, and unexpected worker exits. On timeout it kills and waits for the worker. Dropping the client also stops and reaps the child process.

### Model methods

| Method | Main request fields | Result |
|---|---|---|
| `engine.ping`, `engine.describe` | empty object | worker status, supported methods, devices |
| `init_model` | `config`, `seed` | `engine_id`, step and placement details |
| `forward` | `engine_id`, `batch` | loss and model metrics |
| `train_step` | `engine_id`, `batch`, `seed` | step, loss, gradient norm, elapsed time |
| `causal_ar_init` | AR `config`, `seed`, initialization tokens | `engine_id`, step, maximum length |
| `causal_ar_train_step` | engine, tokens, mask, continuation start; optional teacher | loss, gradient norm, elapsed time |
| `causal_ar_evaluate` | engine, tokens, optional mask, prompt length | NLL, perplexity, token count, token log-likelihoods |
| `generate` | engine, prompts, token limit, temperature, top-k, seed | generated tokens and proposal distributions |
| `speculative_verify` | target engine, prefixes, drafts, proposal distributions, sampling controls, seed | accepted prefix lengths/masks, corrected continuation tokens, acceptance rate |
| `save_checkpoint`, `load_checkpoint` | engine and Rust-selected path | checkpoint path and step |
| `probe.train` | train/evaluation token arrays and probe settings | probe NLL |

The verifier evaluates each prefix plus draft and bonus position in one JAX forward pass. Its target distribution applies the same temperature and top-k controls as the drafter distribution before acceptance and correction sampling.

The Rust integration tests exercise this line-framed contract, including AR initialization, training, evaluation, generation, speculative verification, checkpoint round-trips, and worker error propagation. CI runs those with the full Rust test suite.
