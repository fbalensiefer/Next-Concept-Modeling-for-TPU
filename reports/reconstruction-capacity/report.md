# Fixed-length reconstruction: duration and code capacity

Six fresh CPU fits use the same 128-byte windows, training orders and seeds as paired controls. Codebook size (16 versus 64) is the sole architecture change. All runs use 2,000 updates, batch size four, reconstruction-first losses and the final checkpoint. Validation remains descriptive; test reconstruction is unevaluated. Source snapshots are preserved under the run directory.

| Codebook | Seed | Step | Train byte accuracy | Validation byte accuracy | Validation exact windows | Punctuation accuracy |
|---|---|---|---|---|---|---|
| 16 | 0 | 100 | 80.5806% | 79.6753% | 0.0000% | 23.4440% |
| 16 | 0 | 500 | 95.6250% | 94.8712% | 0.2965% | 81.6151% |
| 16 | 0 | 1000 | 96.1633% | 95.4538% | 0.8895% | 84.4718% |
| 16 | 0 | 2000 | 96.5638% | 95.8980% | 1.1119% | 84.9346% |
| 16 | 1 | 100 | 81.7419% | 80.7091% | 0.0000% | 34.8069% |
| 16 | 1 | 500 | 95.0026% | 94.4386% | 0.1483% | 76.7794% |
| 16 | 1 | 1000 | 96.1348% | 95.4845% | 0.8154% | 79.6042% |
| 16 | 1 | 2000 | 96.4930% | 95.6501% | 1.1861% | 84.5196% |
| 16 | 2 | 100 | 81.5459% | 80.6691% | 0.0000% | 35.2697% |
| 16 | 2 | 500 | 94.8166% | 94.0123% | 0.0000% | 77.2103% |
| 16 | 2 | 1000 | 95.6243% | 94.8810% | 0.2224% | 81.2640% |
| 16 | 2 | 2000 | 96.0381% | 95.2430% | 0.6672% | 81.3438% |
| 64 | 0 | 100 | 88.2164% | 87.4948% | 0.0000% | 49.5372% |
| 64 | 0 | 500 | 99.2095% | 98.8643% | 34.8406% | 94.4143% |
| 64 | 0 | 1000 | 99.4617% | 99.1938% | 44.1809% | 96.9039% |
| 64 | 0 | 2000 | 99.5862% | 99.3311% | 51.3714% | 97.6380% |
| 64 | 1 | 100 | 87.1359% | 86.3539% | 0.0000% | 41.9087% |
| 64 | 1 | 500 | 99.0241% | 98.8406% | 31.9496% | 94.2228% |
| 64 | 1 | 1000 | 99.4154% | 99.2454% | 45.7376% | 96.7124% |
| 64 | 1 | 2000 | 99.5299% | 99.3745% | 53.0764% | 98.1009% |
| 64 | 2 | 100 | 87.0809% | 86.1628% | 0.0000% | 48.8509% |
| 64 | 2 | 500 | 99.2538% | 99.0375% | 36.4715% | 95.4676% |
| 64 | 2 | 1000 | 99.4948% | 99.2981% | 45.6635% | 97.8455% |
| 64 | 2 | 2000 | 99.4055% | 99.1788% | 40.6968% | 97.0156% |

## Paired descriptive effects

- 16 entries, 100 to 2,000 updates: mean validation byte accuracy change 15.25 percentage points; seed range 14.57 to 16.22.
- 64 entries, 100 to 2,000 updates: mean validation byte accuracy change 12.62 percentage points; seed range 11.84 to 13.02.
- 64 versus 16 entries at 2,000 updates: mean paired difference 3.70 percentage points; seed range 3.43 to 3.94.

These ranges describe three model seeds on one fixed validation book. They are not population confidence intervals or independent corpus replications. The follow-up was chosen after observing the initial pilot, so its findings are exploratory.

## Cost and controls

| Representation | Nominal bits/input byte | Reconstruction training byte exposure |
|---|---|---|
| Raw bytes | 8 | 0 |
| Fixed byte pairs | 8 | 0 |
| RVQ, 16 entries | 7 | 1,024,000 per fit |
| RVQ, 64 entries | 10.5 | 1,024,000 per fit |

Both exact controls reconstruct every train/validation window perfectly. Packing byte pairs does not compress them: the output vocabulary has 65,536 categories, whose prediction heads and sampling costs must be charged in any prior comparison. RVQ nominal bits exclude codec weights, codebook tables and framing. More codebook entries change both information capacity and optimization; this experiment does not isolate an information-theoretic cause.

| Codebook | Seed | Codec parameters | Codebook embedding elements | Full run wall seconds |
|---|---|---|---|---|
| 16 | 0 | 21328 | 768 | 35.8 |
| 16 | 1 | 21328 | 768 | 26.6 |
| 16 | 2 | 21328 | 768 | 26.5 |
| 64 | 0 | 21328 | 3072 | 30.3 |
| 64 | 1 | 21328 | 3072 | 29.6 |
| 64 | 2 | 21328 | 3072 | 29.8 |

Wall time includes compilation, full train/validation evaluation every 100 steps and serialization. It is not a matched training-cost or generation-latency measure. No TPU was used.

## Decision

Longer training improves validation byte accuracy in every seed for both codebook sizes. The larger codebook also improves final byte accuracy in every paired seed. These observations support a bounded next check of the strongest codec on independent documents and consequential error patterns. Keep the current codecs experimental; no application fidelity tolerance has been established. Raw and fixed-block controls already offer exact reconstruction without codec training; a learned codec must still demonstrate generation value.

The remaining real-text checks include rare patterns, UTF-8 validity, independent documents, a frozen application fidelity tolerance and trained prior comparisons. No claims about generation quality or acceleration follow from these reconstruction results.

## Reproduction

```sh
.venv312/bin/python examples/reconstruction_capacity.py --output runs/reconstruction-capacity-v1
.venv312/bin/python examples/summarize_reconstruction_capacity.py --run runs/reconstruction-capacity-v1 --output reports/reconstruction-capacity
```

Use a fresh run directory. Requires the initial pilot’s processed files and protocol. [Protocol](protocol.json), [all checkpoint metrics](summary.json), [exact controls](controls.json).

Validation: 12 targeted tests passed, including exhaustive bijectivity over all 65,536 byte pairs, invalid input rejection, split leakage, masked metrics, document-mode training and checkpoint resume. Experiment and report scripts compile; `git diff --check` passes.
