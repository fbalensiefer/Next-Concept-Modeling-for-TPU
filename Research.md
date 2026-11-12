# Research

This project explores hierarchical discrete representations for parallel text generation and representation transfer between generative architectures. Both papers are by Fabian Balensiefer.

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

The priority is information preservation, uncertainty-aware fidelity checks and dependency structure. Production defaults remain unchanged. Both papers share the updated evidence; they are not independent replications.
