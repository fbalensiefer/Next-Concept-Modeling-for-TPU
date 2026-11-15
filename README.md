# Next-Concept Modeling for TPU

A research implementation of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162). Rust owns experiment orchestration, corpus preparation, statistics, and reports. A headless Python/JAX/Flax worker owns model definitions, tensor computation, gradients, optimizer updates, and accelerator execution. Rust communicates with that worker only through JSON-RPC over stdio.

The central research question is whether a learned hierarchy can improve the quality–latency tradeoff for complete text generation, including its decoder. Current CPU and synthetic-domain results do not establish real-text quality or TPU acceleration. See the [experimental contract](docs/experimental-contract.md) for measurement rules, evidence, and open gates.

## Quick start

Install Rust and Python 3.12. Set up the JAX model-engine environment, then use the native Rust CLI for research commands:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export NCM_PYTHON="$PWD/.venv/bin/python"

cargo run --bin ncm-run -- --list
cargo run --bin ncm-run -- engine-info
cargo run --bin ncm-run -- train \
  --config examples/text/config.json \
  --output runs/text-rust
```

`NCM_PYTHON` selects the JAX worker environment and defaults to `python3`. See the [Rust/JAX guide](docs/rust-jax.md) for the IPC contract, [real-text pilot guide](docs/real-text-reconstruction.md), and [benchmark guide](docs/decisive-benchmark.md).

## Research status

The [document-separated real-text pilot](reports/real-text-reconstruction/report.md) records reconstruction errors after short training runs. The [dependency controls](reports/text-dependency-priors/report.md) show local dependency gains without establishing a learned-code generation advantage.

The [Rust real128 benchmark record](reports/validation-controls/real128-rust-native/README.md) covers seeds 200–202: the strict validator accepted 351 timing/reference rows, 45 quality rows and 135 checkpoint observations. This records pipeline completion; no speedup or TPU performance claim follows.

Research papers:

- [NCM systems paper (PDF)](output/pdf/ncm_systems_paper.pdf)
- [SCALE research paper (PDF)](output/pdf/SCALE_research_paper.pdf)
- [Research overview](Research.md)
- [Paper source and build guide](paper/README.md)

## Repository layout

- `rust/`: native command line, orchestration, data and report logic
- `src/ncm/`: JAX/Flax engine modules and the headless JSON-RPC worker
- `src/ncm/workloads/`: historical sources for experiments without a native Rust replacement; not packaged or exposed by the CLI
- `tests/`: Rust contract tests and Python model-engine tests
- `examples/`, `paper/`, and `reports/`: configurations, manuscripts, and evidence

Historical Python research scripts that have native Rust replacements are removed. Remaining legacy research files are not used by `ncm-run`; all supported commands are listed by `cargo run --bin ncm-run -- --list`.

Project by Fabian Balensiefer. The underlying NCM method is due to Kumbong et al.
