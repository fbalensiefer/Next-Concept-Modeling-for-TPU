# Next-Concept Modeling for TPU

A JAX/Flax implementation of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) for TPU-oriented execution. A multiscale codec encodes text into hierarchical discrete codes, and a shared Transformer generates each level in parallel.

The central question is: **can a learned hierarchy make text generation faster while preserving information and modeling the dependencies that matter?**

The immediate focus is a modest real-text experiment comparing raw bytes, fixed blocks and learned RVQ representations, followed by minimal dependency-aware priors and a small end-to-end TPU feasibility check. Success means a better quality–latency tradeoff for the complete generator and decoder. SCALE and pretrained-model conversion are later, conditional extensions.

See the [experimental contract and decision gates](docs/experimental-contract.md) for the next work package, measurement rules, existing evidence and open execution prerequisites. Current CPU and synthetic-domain results do not establish real-text quality or TPU acceleration.

Project by Fabian Balensiefer. The underlying NCM method is due to Kumbong et al.

## Quick start

Python 3.12 is the tested development environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/train.py
```

The [first document-separated real-text pilot](reports/real-text-reconstruction/report.md) records substantial reconstruction errors after short training runs. See the [reproduction instructions](docs/real-text-reconstruction.md). A [paired duration/capacity follow-up](reports/reconstruction-capacity/report.md) tests longer training and larger codebooks against exact representation controls.

The [real-text dependency controls](reports/text-dependency-priors/report.md) compare independent and first-order priors on all three representations; they show local dependency gains without establishing a learned-code generation advantage.

## Research papers

- [NCM systems paper (PDF)](output/pdf/ncm_systems_paper.pdf)
- [SCALE research paper (PDF)](output/pdf/SCALE_research_paper.pdf)

See [Research.md](Research.md) for brief introductions and [paper sources](paper/README.md) for build instructions.

## Explore

[Source](src/ncm/) · [Examples](examples/) · [Documentation](docs/) · [Experiment reports](reports/README.md)

[Citation](CITATION.cff) · [MIT license](LICENSE)
