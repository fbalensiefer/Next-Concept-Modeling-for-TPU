# Architecture boundary audit

Base revision: 34cc7b2.

Rust remains a dependency-free orchestration crate: command dispatch, process execution, paper-build sequencing, and publication of checked PDFs. Model computation, training, quantization, sampling, and numerical research calculations remain in the JAX/Flax package. The earlier duplicate Rust model implementation is absent.

This audit found and removed a remaining Python process-orchestration loop from the paper auditor. Rust now sequences the three evidence checks and the final audit, propagating the first worker failure. Python retains the actual evidence and manuscript checks. A Rust integration regression checks both execution order and early termination.

No obsolete standalone Python entrypoints remain outside `src/ncm` and `tests`. Required research workloads remain packaged under `src/ncm/workloads` for reproducibility. Old bytecode caches in `examples`, `scripts`, and `paper` were removed locally. A stale benchmark command in a workload docstring was corrected. Historical experiment records were preserved.

Both manuscript sources now state the Rust/JAX boundary briefly; the repeated implementation paragraph was removed from the systems paper. Both PDFs were rebuilt and rendered for review.

Validation for this audit: three Rust integration tests; Rust formatting and Clippy with warnings treated as errors; 18 relevant Python checks; both paper evidence/citation/reference/layout-log audits; source and document checksum verification. The complete 121-test, four-device model suite was already verified on the base revision; model-engine calculations did not change in this audit.
