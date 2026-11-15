# Rust real128 benchmark record

Strict suite validation completed on November 15, 2026 for seeds 200, 201 and 202 and prompt lengths 0, 16 and 64.

| Artifact | Rows |
| --- | ---: |
| `drafter_comparison.csv` | 351 |
| `quality_rows.csv` | 45 |
| `checkpoint_audit.csv` | 135 |
| `training_rows.csv` | 45 |

`suite_validation.json` records status `complete`, the expected combinations and signature checks. JSON companions retain structured results. `sweep.json` and `drafter_timing_run.json` preserve configuration and recorded source identities; `sweep_progress.json` records completed timing cells.

The run used base commit `9c4d59b340e2c6c2790ee9c77a2a081654a1f1ce` plus the IPC/progress working-tree changes committed with this report. Training, quality and checkpoint stages were retained from the November 14 execution. Timing resumed and completed on November 15. Source hashes in the original records are preserved rather than retroactively relabelled.

`execution.log`, `execution.jsonl` and the original section of `run_provenance.txt` describe the earlier attempt, which was interrupted. The later `benchmark-drafters-atomic-20261115.log` records a failed startup using system Python without JAX, followed by successful execution with `.venv312/bin/python3`. `execution_status.txt` states the final status. The resumed timing stage still took hours; no controlled before/after speedup is available. Structural validation does not prove numeric identity to historical results or TPU performance.

Revalidate from the repository root:

```sh
cargo run --release --bin ncm-run -- validate-benchmark-suite --sweep reports/validation-controls/real128-rust-native
```

All generated results are isolated here. SHA-256 checks of the historical `real128/drafter_comparison.csv`, `quality.csv` and `checkpoint_curves.csv` matched their pre-run values listed in `run_provenance.txt`. Checkpoints are retained for this small run so the artifact record remains self-contained.
