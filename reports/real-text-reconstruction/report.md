# First document-separated real-text reconstruction pilot

Nine reconstruction-first fits ran on CPU with three model seeds at each byte length. These are 100-update optimization probes, not converged comparisons. The final checkpoint was used; test reconstruction remains unevaluated.

Training document: [Alice’s Adventures in Wonderland](https://www.gutenberg.org/ebooks/11). Validation: [Through the Looking-Glass](https://www.gutenberg.org/ebooks/12). Sealed test: [The Hunting of the Snark](https://www.gutenberg.org/ebooks/13). One author and one document per split severely limit generalization and prevent meaningful document-population confidence intervals.

| Bytes | Seed | Validation byte accuracy | Exact-window accuracy | Punctuation accuracy | Training byte exposure | Run wall seconds |
|---|---|---|---|---|---|---|
| 32 | 0 | 73.00% | 0.0185% | 6.10% | 12800 | 9.3 |
| 32 | 1 | 72.98% | 0.0000% | 15.86% | 12800 | 3.7 |
| 32 | 2 | 70.22% | 0.0000% | 19.55% | 12800 | 4.2 |
| 128 | 0 | 79.68% | 0.0000% | 23.44% | 51200 | 8.4 |
| 128 | 1 | 80.71% | 0.0000% | 34.81% | 51200 | 5.0 |
| 128 | 2 | 80.67% | 0.0000% | 35.27% | 51200 | 5.5 |
| 512 | 0 | 83.99% | 0.0000% | 37.51% | 204800 | 12.7 |
| 512 | 1 | 82.10% | 0.0000% | 31.79% | 204800 | 10.3 |
| 512 | 2 | 82.89% | 0.0000% | 36.68% | 204800 | 9.1 |

## Decision

The codec does not yet demonstrate useful fidelity in this bounded pilot. Near-zero exact reconstruction and weaker punctuation recovery preclude treating aggregate byte accuracy as information preservation. This is not evidence that learned hierarchies cannot work: the training budget is small, and exposure differs across lengths. Before fitting generation priors to these codes, compare a longer reconstruction training curve and a bounded capacity increase at one fixed length, against exact raw/fixed-block reconstruction controls. Preserve this pilot as a negative early reference. No generation-quality or acceleration claim follows.

## Reproduction and limits

See [instructions](../../docs/real-text-reconstruction.md), [protocol](protocol.json), [all metrics](summary.json) and [environment](environment.txt). The protocol records source/data hashes, devices and configuration. The raw corpus and checkpoints remain under ignored `runs/real-text-pilot/`. Re-downloads must match the recorded raw hashes; later Gutenberg revisions are a different dataset. The recorded environment is an installed-package inventory, not a cross-platform resolver lock.

Elapsed time includes compilation, evaluation and checkpoint writes. This is not a generation benchmark. No TPU was available in this local run. Model-seed spread is descriptive; the three seeds are not independent corpus replications. Names, repeated structures and UTF-8 fidelity remain unmeasured. Exact duplicate checks do not exclude near duplicates. Test content was processed only for data identity and overlap checks, not reconstruction evaluation.

Validation: 12 targeted tests passed, covering document isolation, split leakage, masked fidelity metrics, document-mode training, exact checkpoint resume, prior training and benchmark compatibility. `git diff --check` passed.
