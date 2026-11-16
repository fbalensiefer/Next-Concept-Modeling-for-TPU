# Research

This project asks whether a learned hierarchy can accelerate text generation while preserving information and modeling the dependencies that matter. Representation transfer is a later, conditional hypothesis. Both papers are by Fabian Balensiefer.

## NCM systems paper

[A Functional JAX/Flax Implementation of Next-Concept Modeling for TPU-Oriented Execution (PDF)](https://raw.githubusercontent.com/fbalensiefer/Next-Concept-Modeling-for-TPU/main/output/pdf/ncm_systems_paper.pdf)

[LaTeX source](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/paper/ncm_systems_paper.tex)

A practical foundation for studying coarse-to-fine text generation, with mask-aware residual quantization, explicit training state and distributed execution contracts. The paper connects the implementation to reproducible experiments and its design tradeoffs.

## SCALE

[SCALE: Dependency-Aware Representation Compilation from Pretrained Transformers to Hierarchical Generative Models (PDF)](https://raw.githubusercontent.com/fbalensiefer/Next-Concept-Modeling-for-TPU/main/output/pdf/SCALE_research_paper.pdf)

[LaTeX source](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/paper/SCALE_research_paper.tex)

A research framework for adapting pretrained Transformer representations to hierarchical generators. It combines representation alignment with dependency-aware code design and sets out conditional guarantees and experiments for testing the approach.

[Editable sources and build instructions](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/paper/README.md)

## Current evidence and revised research priority

The original coarse-codebook stability result holds within its tested domain. Subsequent [matched ablations](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/reports/maintenance-ablation/report.md) and [360 fresh-seed runs across six binary source laws](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/reports/fresh-domain-validation/report.md) expose failures under every quantizer condition. A [480-fit frozen-codec comparison](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/reports/dependency-validation/report.md) shows practical gains from joint and sparse conditional priors, including a capacity/estimation tradeoff.

[Exact invertible innovation controls](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/docs/invertible-baselines.md) show that Markov and parity benchmarks admit independent coordinates with inexpensive exact inverses. A [causal alignment-target diagnosis](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU/blob/main/reports/alignment-diagnosis/report.md) identifies information absent from the current feature target. These are mechanism studies and classical mathematical controls, not evidence of natural-language quality, universal transfer or TPU speedup.

The bounded unconditional comparison now triggers `halt_quantizer_expansion_and_pivot` for the tested RVQ configuration. Prioritize lossless controls, simple structured priors, and a compute-normalized comparison across independent books before TPU scaling. The [experimental contract](docs/experimental-contract.md) defines the decision gates and remaining prerequisites. Exact invertibility remains a control rather than a universal requirement. Production defaults remain unchanged. Both papers share the updated evidence; they are not independent replications.

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
