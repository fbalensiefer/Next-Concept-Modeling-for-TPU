# Document-safe reconstruction pilot

The bounded pilot uses three distinct Project Gutenberg books (IDs 11, 12 and 13) as train, validation and sealed test documents. This is a single-author feasibility study with one document per split, not representative corpus evaluation. No document-level population uncertainty interval can be justified from one validation document.

Download the UTF-8 versions into `runs/real-text-pilot/raw/{11,12,13}.txt`, then run:

```sh
.venv312/bin/python examples/real_text_reconstruction.py --output runs/real-text-pilot --steps 100
```

Use a fresh output directory for each protocol. The runner strips Gutenberg boundary markers and boilerplate, normalizes CRLF to LF, preserves the body otherwise, and records raw/processed hashes. Source files and package versions are hashed/recorded before training. Source downloads retain the full Gutenberg terms; consult the source's license and jurisdiction restrictions before redistribution. Corpus content and checkpoints stay in ignored `runs/`.

Each run trains a reconstruction-first RVQ codec with entropy weight zero, no feature alignment and no prior loss. Quantization/commitment training remains active. The final checkpoint is reported without validation selection. Seeds 0, 1 and 2 run at 32, 128 and 512 bytes with 100 updates each, batch size four. These are bounded optimization checks, not converged or compute-matched comparisons across lengths: byte exposure increases with length. Elapsed wall time includes initialization, compilation, training, evaluation and serialization, and is not accelerator training time or generation latency.

`documents_jsonl` input contains one `{"id": "unique-id", "text": "..."}` per line. The loader rejects overlapping document identities or exact content across splits, generates nonoverlapping full byte windows inside each document, and drops each incomplete tail. Byte boundaries can split UTF-8 characters. Exact content matching is not near-duplicate detection. The optional test path is checked for split integrity but is never evaluated during training.

Outputs include byte accuracy, exact-window accuracy, per-position error counts, digit and ASCII punctuation slices, per-document sufficient statistics and dropped-tail bytes. Masks exclude padding in the legacy text mode. Names, repeated structures, UTF-8 validity, document bootstrap intervals, and application-specific stress suites remain follow-up work; these metrics do not constitute complete fidelity certification.

This adds document-aware loading to `ncm.experiment` without changing legacy fixture behavior. The neural prior runner now inherits document format from the source codec; the general benchmark runner still uses its legacy file loader and needs migration. See the [experimental contract](experimental-contract.md) for the broader comparisons and [pilot results](../reports/real-text-reconstruction/report.md) for observed limits.
