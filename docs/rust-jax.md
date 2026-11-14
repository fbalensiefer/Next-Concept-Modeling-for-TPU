# Rust orchestration and the JAX engine

Rust owns command dispatch, worker processes, and the paper build sequence. Python/JAX/Flax owns model computation and numerical research workloads. There is one model implementation.

Run commands from the repository or a subdirectory:

```sh
export NCM_PYTHON="$PWD/.venv312/bin/python"
cargo run --bin ncm-run -- --list
cargo run --bin ncm-run -- run-drafter-comparison --help
cargo run --bin ncm-run -- replay-dependency-results
cargo run --bin ncm-run -- build-papers --build-dir /tmp/ncm-papers
```

`NCM_PYTHON` selects the Python environment and defaults to `python3`. `NCM_TECTONIC` selects the paper compiler and defaults to `tectonic`. Paper builds accept `--offline` once the Tectonic resource cache is populated. The launcher forwards worker arguments without invoking a shell, selects the repository working directory, exposes the source package, and propagates worker exit codes.

The former `examples/*.py`, `scripts/*.py`, and `paper/*.py` entrypoints have been removed. Their required calculations are packaged under `src/ncm/workloads/{research,validation,papers}`. Commands use the former filename with underscores replaced by hyphens, for example `run_drafter_comparison.py` becomes `run-drafter-comparison`. `build-papers` replaces the former Python build wrapper with Rust orchestration. Historical reports retain the original commands and source hashes as records of executed experiments.

Configuration and text fixtures remain in `examples/`; manuscript sources and evidence inputs remain in `paper/`. The initial duplicate Rust model primitives were removed because model execution belongs to JAX.
