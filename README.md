# Next-Concept Modeling for TPU

A JAX/Flax implementation of [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) for TPU-oriented execution. A multiscale residual vector quantizer encodes text into hierarchical codes, and a shared Transformer predicts each level's positions in parallel.

Project by Fabian Balensiefer. The underlying NCM method is due to Kumbong et al.

**[Research paper (PDF)](output/pdf/ncm_systems_paper.pdf)**

## Implementation

The core includes mask-aware quantization, functional EMA codebooks, combined reconstruction and prediction losses, runtime codebook freezing, byte fallback routing, and explicit `dp`/`fsdp` sharding. See the paper for the formulation and limitations.

- [Model and training code](src/ncm/)
- [Training example](examples/train.py)
- [Tests](tests/)

## Run

Python 3.12 is the tested development environment. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/train.py
```

## Status

Eleven CPU tests pass, including a virtual four-device sharding check. TPU performance and trained-model quality remain unmeasured. Rust data tooling and a complete byte-stream codec are planned.

[Validation details](paper/validation.json) · [Citation](CITATION.cff) · [MIT license](LICENSE)
