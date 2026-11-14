# Rust orchestration and the JAX engine

Rust owns command dispatch, corpus preparation, report provenance and serialization, FLOP estimates, categorical statistics, dependence metrics, token accounting, and the migrated codec-training runner. Python/JAX/Flax owns model definitions, tensor computations, gradients, optimizer updates, and accelerator placement. There is one model implementation.

Run commands from the repository or a subdirectory:

```sh
export NCM_PYTHON="$PWD/.venv312/bin/python"
cargo run --bin ncm-run -- --list
cargo run --bin ncm-run -- train --config examples/text/config.json --output runs/text-rust
cargo run --bin ncm-run -- real-text-reconstruction --output runs/real-text-pilot-rust --steps 100
cargo run --bin ncm-run -- replay-dependency-results
cargo run --bin ncm-run -- build-papers --build-dir /tmp/ncm-papers
cargo run --bin ncm-run -- generate-bibliographies --check
cargo run --bin ncm-run -- audit-papers
```

`NCM_PYTHON` selects the Python environment and defaults to `python3`. `NCM_TECTONIC` selects the paper compiler and defaults to `tectonic`. Paper builds accept `--offline` once the Tectonic resource cache is populated. The training command reads text or document JSONL inputs in Rust, and sends batches, optimizer steps, evaluation requests, and checkpoint operations through the JSON-RPC worker. Its manifest records input hashes and implementation provenance.

Legacy Python scripts remain in the repository for output-parity checks. Codec training, the bounded real-text reconstruction pilot, the synthetic TC probe, maintenance and domain summaries, dependency-prior summaries, source-law baselines, alignment diagnostics, reconstruction-capacity reporting, paper result generation, bibliography generation, paper auditing, and manifest refreshes run in Rust. The decisive benchmark, drafter comparison, and other historical diagnostics still use the compatibility launcher while their host protocols and JAX worker operations are migrated.

Configuration and text fixtures remain in `examples/`; manuscript sources and evidence inputs remain in `paper/`. The initial duplicate Rust model primitives were removed because model execution belongs to JAX.

Rust library modules `corpus`, `evaluation`, `accounting`, `dependence`, `flops`, `table_prior`, and `validation_io` provide native host-side data, metrics, statistics, and report implementations. Python files are retained in this phase.

## Model-engine IPC

Rust starts one `ncm.engine_rpc` process and exchanges one JSON-RPC 2.0 request and response per line over stdin/stdout. Diagnostics belong on stderr. The worker supports model initialization, forward passes, training steps, checkpoint save/load, and conditional probe fitting, in addition to `engine.ping` and `engine.describe`. Tensor payloads use typed Rust request structs and JSON arrays. Specialized research commands are being moved onto this boundary while Python remains the model engine.

Run `cargo run --bin ncm-run -- engine-info` to check that Rust can start the configured JAX/Flax engine and query its devices.
