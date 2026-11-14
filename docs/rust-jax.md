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

`NCM_PYTHON` selects the Python environment and defaults to `python3`. `NCM_RPC_TIMEOUT_MS` sets a per-request worker timeout (default 300000 ms). `NCM_TECTONIC` selects the paper compiler and defaults to `tectonic`. Paper builds accept `--offline` once the Tectonic resource cache is populated. The training command reads text or document JSONL inputs in Rust, and sends batches, optimizer steps, evaluation requests, and checkpoint operations through the JSON-RPC worker. Its manifest records input hashes and implementation provenance.

Legacy Python scripts remain in the repository for output-parity checks and JAX-internal tests. Codec training, the bounded real-text reconstruction pilot, the synthetic TC probe, maintenance and domain summaries, dependency-prior summaries, source-law baselines, alignment diagnostics, reconstruction-capacity reporting, paper result generation, bibliography generation, paper auditing, manifest refreshes, the decisive causal-AR evaluation, and drafter comparison run in Rust. The two benchmark commands use the JSON-RPC worker directly; they no longer dispatch through Python compatibility launchers.

Configuration and text fixtures remain in `examples/`; manuscript sources and evidence inputs remain in `paper/`. The initial duplicate Rust model primitives were removed because model execution belongs to JAX.

Rust library modules `corpus`, `evaluation`, `accounting`, `dependence`, `flops`, `table_prior`, and `validation_io` provide native host-side data, metrics, statistics, and report implementations. Python files are retained in this phase.

## Model-engine IPC

Rust starts one `ncm.engine_rpc` process and exchanges one JSON-RPC 2.0 request and response per line over stdin/stdout. Diagnostics belong on stderr. The worker supports NCM initialization, training and forward evaluation; causal AR initialization, training, conditioned generation, masked likelihood evaluation and batched speculative verification; checkpoint save/load; and conditional probe fitting. Tensor payloads use typed Rust request structs and JSON arrays. The decisive benchmark batches evaluation data and computes token-loss summaries in Rust. The drafter comparison coordinates generation and verification, then aggregates acceptance, throughput, latency, and paired throughput statistics in Rust. Their outputs are JSON, CSV, and Markdown as applicable.

Run `cargo run --bin ncm-run -- engine-info` to check that Rust can start the configured JAX/Flax engine and query its devices.
