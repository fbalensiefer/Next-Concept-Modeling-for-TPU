# Research

This project explores hierarchical discrete representations for parallel text generation and representation transfer between generative architectures. Both papers are by Fabian Balensiefer.

## NCM systems paper

[A Functional JAX/Flax Implementation of Next-Concept Modeling for TPU-Oriented Execution (PDF)](output/pdf/ncm_systems_paper.pdf)

A practical foundation for studying coarse-to-fine text generation, with mask-aware residual quantization, explicit training state and distributed execution contracts. The paper connects the implementation to reproducible experiments and its design tradeoffs.

## SCALE

[SCALE: Dependency-Aware Representation Compilation from Pretrained Transformers to Hierarchical Generative Models (PDF)](output/pdf/SCALE_research_paper.pdf)

A research framework for adapting pretrained Transformer representations to hierarchical generators. It combines representation alignment with dependency-aware code design and sets out conditional guarantees and experiments for testing the approach.

[Editable sources and build instructions](paper/README.md)

## Current evidence and revised research priority

The original coarse-codebook stability result holds within its tested domain. Subsequent [matched ablations](reports/maintenance-ablation/report.md) and [360 fresh-seed runs across six binary source laws](reports/fresh-domain-validation/report.md) expose failures under every quantizer condition. A [480-fit frozen-codec comparison](reports/dependency-validation/report.md) shows practical gains from joint and sparse conditional priors, including a capacity/estimation tradeoff.

[Exact invertible innovation controls](docs/invertible-baselines.md) show that Markov and parity benchmarks admit independent coordinates with inexpensive exact inverses. A [causal alignment-target diagnosis](reports/alignment-diagnosis/report.md) identifies information absent from the current feature target. These are mechanism studies and classical mathematical controls, not evidence of natural-language quality, universal transfer or TPU speedup.

The priority is information preservation, uncertainty-aware fidelity checks and dependency structure. Production defaults remain unchanged. Both papers share the updated evidence; they are not independent replications.
