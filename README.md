# Next-Concept Modeling for TPU

A JAX/Flax implementation of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) for TPU-oriented execution. A multiscale codec encodes text into hierarchical discrete codes, and a shared Transformer generates each level in parallel.

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

## Research papers

- [NCM systems paper (PDF)](output/pdf/ncm_systems_paper.pdf)
- [SCALE research paper (PDF)](output/pdf/SCALE_research_paper.pdf)

See [Research.md](Research.md) for brief introductions and [paper sources](paper/README.md) for build instructions.

## Explore

[Source](src/ncm/) · [Examples](examples/) · [Documentation](docs/) · [Experiment reports](reports/README.md)

[Citation](CITATION.cff) · [MIT license](LICENSE)
