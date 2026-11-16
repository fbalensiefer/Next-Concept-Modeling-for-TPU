# Research papers

Both papers are authored by Fabian Balensiefer. The November 12, 2026 research extension preserves the historical baseline and adds the same shared evidence to both papers. A November 16 note records the Phase 2 categorical RVQ prior and its bounded Rust generation telemetry pilot in that shared findings section:

- [Systems paper](../output/pdf/ncm_systems_paper.pdf), [editable source](ncm_systems_paper.tex): functional JAX/Flax contracts, historical engineering/CPU results, and current maintenance and dependency studies.
- [SCALE paper](../output/pdf/SCALE_research_paper.pdf), [editable source](SCALE_research_paper.tex): conditional compiler analysis, broader failure boundaries, empirical dependency controls, exact innovation baselines and the alignment-target diagnosis.

The Phase 2 record distinguishes the level-major causal categorical prior from the original parallel shared-level NCM predictor. Its three training steps and single prompted sample verify the complete codec/prior/Rust path; they do not measure held-out generation quality.

They describe shared experiments, not independent replications. The historical 20-model stability success remains valid within its domain. The later 360-run study finds failures under every quantizer condition; 480 training-only fits on frozen reference codecs demonstrate dependency-model improvements. No pretrained transfer, natural-language quality, total adaptation savings or TPU performance is established. Production defaults remain unchanged.

The November 15 Rust-orchestrated real128 run is described concisely in the shared findings included by both papers. Its [run record](../reports/validation-controls/real128-rust-native/README.md) documents complete structural validation and the limits of the evidence.

## Build and check

Use Rust, Python 3.11 or newer, and Tectonic with the packages in `shared_preamble.tex`. Set `NCM_PYTHON` to the project Python environment and `NCM_TECTONIC` if Tectonic is not on PATH:

```sh
cargo run --bin ncm-run -- build-papers --build-dir /tmp/ncm-paper-build
cargo run --bin ncm-run -- generate-results --check
cargo run --bin ncm-run -- generate-recent-results --check
cargo run --bin ncm-run -- generate-bibliographies --check
cargo run --bin ncm-run -- audit-papers --build-dir /tmp/ncm-paper-build
```

The build regenerates evidence and bibliography inputs, checks references/logs, then copies final PDFs to `output/pdf`. Tectonic can use `--offline` after its package cache is populated. Render both complete PDFs with Poppler and inspect every page before publishing. Keep logs and page images outside the repository.

## Evidence and provenance

- `shared_experiments.tex` retains historical observations; `shared_recent_findings.tex` adds the new studies and mathematical controls. Both papers include them.
- `cargo run --bin ncm-run -- generate-results --check` validates historical evidence and the original 44-test record. The archived experiment commit must match its recorded source hashes; current model code is checked separately. It does not assert that today's entire package is the old snapshot.
- `cargo run --bin ncm-run -- generate-recent-results --check` derives current tables from raw archived results, checks study counts and innovation/alignment invariants, and hashes its inputs in `generated/recent_results.json`.
- `references.json` supplies both generated citation subsets. Classical dependence trees and associative scan are credited; innovation controls are not claimed as novel algorithms.
- `validation.json` preserves the historical 44-test verification. `validation_current.json` records the November 12 research-extension checks: 62 Python tests passed and the public 480-fit replay succeeded. Current Phase 3 verification is recorded in `reports/validation-controls/phase3_validation.json`.
- `source_manifest.sha256` identifies current implementation, tests and examples. `document_manifest.sha256` includes manuscripts, all reports, compressed replay inputs and both PDFs. Refresh with `cargo run --bin ncm-run -- update-manifests` after final build/review, then verify with `shasum -a 256 -c`.
- Historical protocol hashes and new study continuation records identify executed versions; later diagnostic fixes are not retroactively described as training changes. The float32 verification interruption and one missing timing remain documented.

Public replay does not require ignored checkpoints:

```sh
cargo run --bin ncm-run -- summarize-dependency-results
cargo run --bin ncm-run -- alignment-diagnosis
cargo run --bin ncm-run -- domain-source-baselines
cargo run --bin ncm-run -- summarize-maintenance
cargo run --bin ncm-run -- summarize-domain-validation --from-reports
```

Model checkpoints stay under ignored `runs/`; source laws and compressed public inputs support independent re-evaluation. The old text fixture logs remain in `reports/text-experiments` and show overfitting, not a successful held-out language result. Native Rust report generators are the supported path; historical extraction scripts are retained only where no native replacement exists.

See [consistency audit](consistency_audit.md). No archival DOI is assigned.

The editable manuscripts now include the Phase 2 unconditional benchmark table and stop decision through `shared_recent_findings.tex`. The checked-in PDFs predate this documentation update and have not been rebuilt. See the [benchmark summary](../reports/unconditional-benchmark/benchmark_summary.md) for the recorded measurements and limitations.
