# Research manuscript

`ncm_systems_paper.tex` is the self-contained LaTeX source for **A Functional
JAX/Flax Implementation of Next-Concept Modeling for TPU-Oriented Execution**,
authored by **Fabian Balensiefer**, dated November 11, 2026.

The bibliography and vector architecture figure are embedded in the source.
No external `.bib` file, image, university template, or proprietary font is
required. Build with a standard LaTeX installation:

```sh
cd paper
pdflatex ncm_systems_paper.tex
pdflatex ncm_systems_paper.tex
```

Alternatively, run `tectonic ncm_systems_paper.tex`. A verified PDF preview is
provided in `output/pdf/ncm_systems_paper.pdf`, relative to the repository root.

The paper distinguishes the original NCM method from this repository's
implementation decisions. It reports the observed eleven passing CPU tests
and synthetic execution check. It does not claim trained-model quality,
measured TPU performance, implemented Rust tooling, or a complete lossless byte
codec.

`source_manifest.sha256` identifies the implementation files inspected for
the manuscript. Paths in the manifest are relative to the repository root.
`validation.json` records the validation environment and observed test outcome.
The manuscript cites the public project repository at
[fbalensiefer/Next-Concept-Modeling-for-TPU](https://github.com/fbalensiefer/Next-Concept-Modeling-for-TPU).
Source hashes identify the described implementation independently of future
updates to its default branch. An archival DOI has not been assigned.

The thirteen bibliography entries cover NCM, VQ-VAE, SoundStream, Transformers,
GShard, JAX, Flax, XLA, Optax, JAX sharding and manual parallelism, Flax lifted
scans, and this implementation. External citation details were checked
against primary papers and official project documentation.
