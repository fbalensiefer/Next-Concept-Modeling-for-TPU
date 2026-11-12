# Exact invertible innovation controls

The Markov and noisy-parity laws admit independent innovation coordinates with exact deterministic inverses. Exhaustive enumeration of all 256 length-eight strings confirms bijectivity, reconstruction and factorization within 1e-14. These classical constructions are newly applied project baselines, not novel algorithms.

`examples/domain_source_baselines.py` regenerates `results.json` from analytic definitions. Tests verify reconstruction of every input and the complete transformed law. No teacher, codec or probability estimator is fitted. Known source parameters are used, so the result is not a finite-sample learning comparison.

The construction and its limits are explained in `docs/invertible-baselines.md`. The important research implication is that the current synthetic laws do not inherently require a lossy residual hierarchy; learning the appropriate coordinates is a separate problem. General language modeling and efficient discovery of such coordinates remain untested.
