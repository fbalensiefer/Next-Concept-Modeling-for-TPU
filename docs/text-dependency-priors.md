# Real-text dependency controls

The stationary table experiment compares independent marginals with sparse first-order Markov models on raw bytes, fixed byte pairs and three frozen learned-code hierarchies. It uses the same document-safe 128-byte windows as the reconstruction study. Each learned-code level is fitted separately; positions share statistics, and cross-level conditioning is omitted. These controls are weaker than the existing neural coarse-conditioned prior.

Every marginal receives total symmetric pseudo-count one. Transition rows receive pseudo-count one distributed according to the fitted positive marginal. An unseen context backs off to that marginal. The sampler draws exactly from this mixture rather than materializing a 65,536-by-65,536 pair transition matrix. No transitions cross window or level boundaries. All smoothing choices are fixed before fitting; validation is used only to report loss. Losses are in nats per code and may be compared between priors within a representation, not interpreted as autoregressive byte likelihood for learned codes.

Run with the initial processed corpus and completed capacity-study checkpoints:

```sh
cargo run --bin ncm-run -- text-dependency-priors --output runs/text-dependency-priors-v1
cargo run --bin ncm-run -- summarize-text-dependency-priors --run runs/text-dependency-priors-v1 --output reports/text-dependency-priors
```

Use a fresh run directory. Exact source snapshots, data/checkpoint hashes and protocol precede fitting. Decoding code sequences must reproduce each frozen codec’s recorded validation byte accuracy. All checkpoint hashes must remain unchanged through completion. The public report preserves compact metrics, raw timings, sample excerpts and a compressed lossless byte archive.

Sample diagnostics include byte-bigram discrepancy, distinct windows, strict UTF-8 validity, adjacent byte repetition and exact training-window matches. They are not a quality evaluation. Reference UTF-8 validity is reported because fixed-byte windows can cut multibyte characters at their boundaries. Three sampling seeds measure sampling variation; they are not three independent corpus replications. Learned-codec seeds remain distinct.

CPU timings include table sampling, all representation conversion, learned decoding where applicable and host output. Python execution and nonoptimized level scheduling prevent extrapolation to TPU speed. Numeric prior payload excludes Python object overhead and codec/device memory; it is not peak memory. A credible quality-matched cached neural autoregressive comparison remains required.

The neural `ncm.prior_experiment` runner now inherits `documents_jsonl` from its source codec, preserves document windows, and checks optional test-split overlap without evaluating test reconstruction. It rejects unknown formats rather than silently reading serialized JSON as text. Legacy plain-text fixture behavior remains available. The general benchmark runner still uses its legacy file loader and must be migrated before a document-corpus neural benchmark.

[Observed results and decision](../reports/text-dependency-priors/report.md)
