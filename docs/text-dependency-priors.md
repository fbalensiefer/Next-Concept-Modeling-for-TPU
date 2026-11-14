# Real-text dependency controls

The stationary table experiment compares independent marginals with sparse first-order Markov models on raw bytes, fixed byte pairs and three frozen learned-code hierarchies. It uses the same document-safe 128-byte windows as the reconstruction study. Each learned-code level is fitted separately; positions share statistics, and cross-level conditioning is omitted. These controls are weaker than the existing neural coarse-conditioned prior.

Every marginal receives total symmetric pseudo-count one. Transition rows receive pseudo-count one distributed according to the fitted positive marginal. An unseen context backs off to that marginal. The sampler draws exactly from this mixture rather than materializing a 65,536-by-65,536 pair transition matrix. No transitions cross window or level boundaries. All smoothing choices are fixed before fitting; validation is used only to report loss. Losses are in nats per code and may be compared between priors within a representation, not interpreted as autoregressive byte likelihood for learned codes.

The completed run and its raw outputs are preserved in
[`reports/text-dependency-priors`](../reports/text-dependency-priors/). The
former Python experiment runner is not part of the supported Rust CLI and is
excluded from the installed package. Repeating this model-backed table-prior
experiment requires a future native Rust workload port.

Sample diagnostics include byte-bigram discrepancy, distinct windows, strict UTF-8 validity, adjacent byte repetition and exact training-window matches. They are not a quality evaluation. Reference UTF-8 validity is reported because fixed-byte windows can cut multibyte characters at their boundaries. Three sampling seeds measure sampling variation; they are not three independent corpus replications. Learned-codec seeds remain distinct.

CPU timings include table sampling, all representation conversion, learned decoding where applicable and host output. The recorded historical runner used Python/JAX orchestration and its timings do not establish TPU speed. Numeric prior payload excludes Python object overhead and codec/device memory; it is not peak memory. A credible quality-matched cached neural autoregressive comparison remains required.

The historical neural `ncm.prior_experiment` runner inherited `documents_jsonl` from its source codec, preserved document windows, and checked optional test-split overlap without evaluating test reconstruction. Current Rust CLI commands and the worker interface are listed in the [Rust/JAX guide](rust-jax.md).

[Observed results and decision](../reports/text-dependency-priors/report.md)
