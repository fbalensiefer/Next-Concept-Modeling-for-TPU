# Next-Concept Modeling for TPU

A research implementation of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162). Rust owns experiment orchestration, corpus preparation, statistics, and reports. A headless Python/JAX/Flax worker owns model definitions, tensor computation, gradients, optimizer updates, and accelerator execution. Rust communicates with that worker only through JSON-RPC over stdio.

The central research question is whether a learned hierarchy can improve the quality–latency tradeoff for complete text generation, including its decoder. Current CPU and synthetic-domain results do not establish real-text quality or TPU acceleration. See the [experimental contract](docs/experimental-contract.md) for measurement rules, evidence, and open gates.

## Quick start

Install Rust and Python 3.11 or newer. Set up the JAX model-engine environment, then use the native Rust CLI for research commands:

```sh
python3.11 -m venv .venv
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

The two-stage RVQ path first screens a reconstruction codec, then trains a causal Transformer over its level-major discrete indices with `freeze_codec=true`. The selected 512-byte seed-0 codec has byte error rate 0.16011568099171897. A three-step prior run and one Rust generation run verify checkpoint loading, prompt conditioning, sampling telemetry and codec decoding; they are pipeline checks, not generation-quality results. See the [RVQ prior and generation guide](docs/rvq-prior-generation.md) and its [recorded metrics](reports/generation-training/checkpoints/prior_metrics.json).

Run the Phase 1 checkpoint gate with `python -m ncm.checkpoint_selection`; it validates matching checkpoint trees strictly and writes `reports/generation-pilot/checkpoint_selection_report.json`. The guide documents the Phase 2 trainer and `ncm-run generate-pilot --mode rvq-prior` arguments and artifacts.

The [Rust real128 benchmark record](reports/validation-controls/real128-rust-native/README.md) covers seeds 200–202: the strict validator accepted 351 timing/reference rows, 45 quality rows and 135 checkpoint observations. This records pipeline completion; no speedup or TPU performance claim follows.

Research papers:

- [NCM systems paper (PDF)](output/pdf/ncm_systems_paper.pdf)
- [SCALE research paper (PDF)](output/pdf/SCALE_research_paper.pdf)
- [Research overview](Research.md)
- [Paper source and build guide](paper/README.md)

## Phase 2 bounded unconditional benchmark

The [recorded benchmark](reports/unconditional-benchmark/benchmark_summary.md) at implementation commit `ea44c2c` compares unconditional 512-byte generation with the strictly restored, frozen seed-0 codec. All neural priors start from fresh weights and receive 100 updates. The recorded decision is **`halt_quantizer_expansion_and_pivot`**: no learned RVQ candidate improves all three measured quality–cost coordinates over both lossless controls.

| Candidate | Held-out BPB / bound | Bigram JSD | Recorded total ms/sequence | Sequential rounds / draws |
|---|---:|---:|---:|---:|
| Raw Byte AR | 4.93 | 0.555 | 3.99 | 512 |
| Reversible XOR AR | 6.38 | 1.000 | 3.91 | 512 |
| RVQ Level-Parallel | 7.23 | 0.280 | 5.63 | 3 levels |
| RVQ Chow–Liu | 9.92 | 0.308 | 0.33 | See diagnostic caveat below |
| RVQ Full-AR | 7.60 | 0.253 | 6.87 | 896 codes |

The frozen codec achieves **83.90% byte accuracy but 0% exact window reconstruction** on 66 test windows. This substantial reconstruction error contributes to the RVQ likelihood bound; better surface bigram fits (JSD 0.253–0.308) do not establish better sequence likelihood. Reconstruction and prior error were not separately ablated, so the codec cannot be identified as the sole cause. Raw/XOR figures are model NLLs; RVQ figures are upper bounds that add code-prior NLL and decoder cross-entropy. Comparing these bounds does not prove dominance in exact marginal likelihood.

One distinct Gutenberg book supplies each of train, validation and test. Non-overlapping test windows avoid training-book leakage, but the reported window standard errors do not measure variation between books. Updates, batch size and transformer dimensions are matched; parameter counts and FLOPs are not. Timings are the recorded batch-amortized CPU measurements, with compilation and warm-up recorded separately. The Chow–Liu diagnostic omits cross-level conditioning; its reported three levels are not a measured tree dependency depth. These results support pausing quantizer expansion for this configuration, not a universal rejection of learned representations or a TPU performance claim.

The next priority is a corrected, compute-normalized comparison of lossless controls and simple structured priors across more independent documents, before any TPU scaling decision.

## Repository layout

- `rust/`: native command line, orchestration, data and report logic
- `src/ncm/`: JAX/Flax engine modules and the headless JSON-RPC worker
- `src/ncm/workloads/`: historical sources for experiments without a native Rust replacement; not packaged or exposed by the CLI
- `tests/`: Rust contract tests and Python model-engine tests
- `examples/`, `paper/`, and `reports/`: configurations, manuscripts, and evidence

Historical Python research scripts that have native Rust replacements are removed. Remaining legacy research files are not used by `ncm-run`; all supported commands are listed by `cargo run --bin ncm-run -- --list`.

Project by Fabian Balensiefer. The underlying NCM method is due to Kumbong et al.
