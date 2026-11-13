# Real-text representation and dependency diagnostics

Ten stationary table fits compare raw bytes, reversible byte pairs and three frozen 64-entry RVQ codecs. Every representation is fitted with an independent marginal and a sparse first-order Markov prior. Each RVQ level has its own prior and no cross-level conditioning. These controls are deliberately weaker than the project’s neural coarse-conditioned prior; they test local dependence rather than replace that model.

Training and validation reuse the book-separated 128-byte windows from the reconstruction experiments. Smoothing strength is fixed at one before fitting. All three learned-codec seeds are retained. No test reconstruction or generation evaluation is performed.

## Held-out modeling loss

Compare loss only between priors for the same representation and codec. RVQ code loss is not a byte likelihood. Even the reversible fixed-pair loss describes a different restricted model family; lower loss alone does not establish better generation.

| Representation | Independent NLL/code | Markov NLL/code | Reduction |
|---|---|---|---|
| raw_bytes | 3.2675 | 2.4075 | 0.8600 |
| fixed_pairs | 5.7091 | 4.1071 | 1.6020 |
| rvq_seed_0 | 4.0229 | 3.5950 | 0.4279 |
| rvq_seed_1 | 3.9859 | 3.5398 | 0.4461 |
| rvq_seed_2 | 4.0361 | 3.6181 | 0.4180 |

## Decoded sample diagnostics

Each cell averages three sampling seeds, 256 windows each. Seeds quantify Monte Carlo variation, not independent data/model fits. The raw and fixed-pair fits are deterministic given this training corpus. Bigram TV is a local distributional diagnostic; coherent language, diversity and semantic quality cannot be inferred from it alone.

| Representation | Prior | Byte-bigram TV ↓ | Distinct windows | Valid UTF-8 windows | Equal adjacent bytes | Exact training-window matches |
|---|---|---|---|---|---|---|
| raw_bytes | independent | 0.4972 | 1.0000 | 0.0000 | 0.0606 | 0.0000 |
| raw_bytes | markov | 0.0880 | 1.0000 | 0.8932 | 0.0272 | 0.0000 |
| fixed_pairs | independent | 0.2588 | 1.0000 | 0.0052 | 0.0454 | 0.0000 |
| fixed_pairs | markov | 0.0894 | 1.0000 | 0.8568 | 0.0271 | 0.0000 |
| rvq_seed_0 | independent | 0.5132 | 1.0000 | 0.1510 | 0.0383 | 0.0000 |
| rvq_seed_0 | markov | 0.4082 | 1.0000 | 0.1628 | 0.0196 | 0.0000 |
| rvq_seed_1 | independent | 0.5251 | 1.0000 | 0.0977 | 0.0424 | 0.0000 |
| rvq_seed_1 | markov | 0.4238 | 1.0000 | 0.0378 | 0.0221 | 0.0000 |
| rvq_seed_2 | independent | 0.5318 | 1.0000 | 0.0716 | 0.0359 | 0.0000 |
| rvq_seed_2 | markov | 0.4472 | 1.0000 | 0.0521 | 0.0182 | 0.0000 |

Validation reference: 90.8080% of its 1349 fixed-byte windows are valid UTF-8. This establishes the boundary-cut limitation independently of generation.

## Local CPU cost

Generation timing includes NumPy random sampling, all code levels, representation conversion, device transfer, neural decoding when applicable and host materialization of 16 complete 128-byte windows. One warmup precedes seven timed batches. RVQ decoder compilation is already incurred during reconstruction auditing, so warmup is not a compilation measure. Raw and pair decoders are exact cheap transforms. These Python table implementations do not establish accelerator latency, peak memory, or speed relative to a cached neural AR baseline.

| Representation | Prior | Fit seconds | Median batch ms | p90 batch ms | Numeric prior payload bytes |
|---|---|---|---|---|---|
| raw_bytes | independent | 0.003 | 0.091 | 0.094 | 4096 |
| raw_bytes | markov | 0.101 | 3.347 | 3.412 | 23216 |
| fixed_pairs | independent | 0.003 | 0.090 | 0.103 | 1048576 |
| fixed_pairs | markov | 0.082 | 1.974 | 2.055 | 1276944 |
| rvq_seed_0 | independent | 0.008 | 1.378 | 1.477 | 3072 |
| rvq_seed_0 | markov | 0.181 | 7.086 | 7.315 | 156896 |
| rvq_seed_1 | independent | 0.008 | 1.288 | 1.542 | 3072 |
| rvq_seed_1 | markov | 0.183 | 7.046 | 7.287 | 141088 |
| rvq_seed_2 | independent | 0.008 | 1.397 | 1.459 | 3072 |
| rvq_seed_2 | markov | 0.189 | 7.175 | 7.424 | 157072 |

Numeric payload excludes Python-object overhead, sparse dictionary counters, codec parameters, codebooks and device/compiler buffers; it is not total or peak memory. Fit time excludes codec training and includes held-out loss computation in this runner. Original codec training costs remain in the capacity study. The stationary first-order model introduces a dependency chain; this implementation samples each level sequentially rather than optimizing parallel execution across levels. No quality–latency success claim follows.

## Decision

Adding adjacent dependencies reduces held-out prior loss in all five representations. Raw and fixed-pair Markov models have substantially lower byte-bigram discrepancy than the learned-code Markov models in this study. Inspected samples remain visibly incoherent; this is an informal observation, not a blinded quality rating. RVQ therefore has no demonstrated generation advantage here. Local dependence is a useful next modeling control, but this study does not select a winning representation or establish that RVQ earns its codec training and information cost. A stronger next comparison needs coarse-conditioned neural learned-code priors, credible cached autoregressive raw-byte generation and a prespecified quality evaluation before selecting a speed operating point.

One training book and one validation book, stationary pooling across positions, finite sampling, and lack of independent document replications limit the evidence. Exact UTF-8 window validity also depends on arbitrary byte cuts; compare against reference validity rather than interpreting every incomplete boundary as a modeling failure. No scalable total-correlation estimate is needed to observe these controlled dependency effects.

## Reproduction and audit

```sh
.venv312/bin/python examples/text_dependency_priors.py --output runs/text-dependency-priors-v1
.venv312/bin/python examples/summarize_text_dependency_priors.py --run runs/text-dependency-priors-v1 --output reports/text-dependency-priors
```

Requires the initial processed corpus and all three capacity-study 64-entry checkpoints. Use a fresh run directory. [Protocol](protocol.json) records source/data/checkpoint hashes and software versions; exact executable sources are saved under the run directory. [All metrics and timings](results.json), [sample excerpts](sample_excerpts.json), [lossless sampled bytes](samples.npz), [frozen-codec integrity](integrity.json). Code-to-byte decoding was checked against all three recorded validation reconstruction accuracies before fitting; checkpoint hashes were checked again after every fit completed.

Validation: 14 targeted tests passed, including smoothed-row normalization, unseen-context backoff, a known alternating dependency, deterministic sampling, exhaustive byte-pair bijectivity, document-safe prior fitting and source checkpoint integrity. All 30 archived sample batches match their original generated bytes; saved executable source hashes and frozen codec hashes were verified. `git diff --check` passes.
