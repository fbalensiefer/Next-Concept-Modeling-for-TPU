# Rust/JAX repository cleanup

Base revision: ba3e7d0ea9a29e8528b4c0763b81e94d56ae4c6e.

Rust dispatches named commands, starts packaged Python workers without a shell, forwards arguments and exit codes, and orchestrates the paper build. Python/JAX/Flax remains the model engine. Twenty-six former standalone Python files were moved into `src/ncm/workloads`; the old standalone locations and Python paper-build wrapper were removed. Historical experiment outputs retain their original provenance and commands.

The duplicate Rust model primitives and their dependencies were removed. The Rust crate now has no external dependencies. Configuration/data fixtures remain in `examples/`. Current instructions use Rust commands; `docs/rust-jax.md` explains the mapping. Both manuscript sources and both published PDFs were updated with brief implementation wording.

Verification:

- All 121 Python tests passed in 236.49 seconds with four virtual CPU devices, including the prior regression and powered verifier tests.
- Two Rust integration tests passed, covering module dispatch, argument boundaries (including spaces and shell punctuation), source-package discovery, repository working directory, worker exit codes, and invalid commands.
- Rust formatting and Clippy passed with warnings treated as errors. Packaged Python files passed syntax compilation.
- The Rust-launched public replay verified all 480 archived fits. The benchmark help command loaded its packaged module successfully.
- Both PDFs passed the existing evidence/citation/reference/log audits; all 34 rendered pages were reviewed. There were no unresolved references, missing glyphs, or overfull boxes.
- The historical paper-source check was repaired to validate the archived 1d600d7 experiment source against its recorded hashes, rather than comparing today's source with historical hashes. No archived results or validation thresholds were changed.

No new training or performance experiment was run. This organization does not establish a speedup or TPU result.
