# Matched maintenance ablation and fixed binary follow-up

The finite-domain comparison rejects ordinary EMA without recovery as a robust replacement. EMA with recovery is a simpler stable candidate. Fixed coarse binary endpoints also succeed on these seeds, but no general quantizer superiority is established.

## Scope and controls

Two sequential CPU studies trained 100 models in total: 60 for three maintenance conditions, followed by 40 for recovery versus fixed binary endpoints. Both studies used student seeds 13–17 and all four existing objective arms. Each model received 400 codec updates and 200 decoder-only replay updates. No teacher or prior was retrained. These are exploratory reuses of previously studied seeds, not fresh confirmation seeds.

The teacher, 16-sequence population, sampled data, neural architecture, parameters, K-means++ initialization, minibatch order, optimizer, loss coefficients, entropy regularization, temperature and commitment schedules were matched. Initial state hashes agree within each seed across treatments. Keeping objectives matched means this is an ablation of centroid movement/recovery, not an ablation of the complete regularization package.

- Minimal EMA: ordinary full-rate EMA centroids, no recovery.
- EMA + recovery: ordinary full-rate EMA centroids and the existing dead-code replacement mechanism.
- Reference: existing full package, including centroid displacement scale 0.25 and recovery.
- Fixed binary: coarse endpoints stay at the identical K-means initialization; encoder/decoder train normally and fine codebooks retain EMA. Two-vector nearest-neighbor assignment is exactly an affine scalar threshold. The executable retains existing nearest-neighbor lookup; it is not a custom scalar kernel or full multilevel FSQ implementation.

Population evaluation enumerates every input and all 32 codewords. Exact conditional marginals define the factorized oracle, which is pushed through the trained deterministic decoder. This isolates code-factorization quality from prior fitting. The oracle uses population probabilities for diagnosis only; training uses the archived observations.

## Results after matched decoder replay

| Condition | Mean sequence reconstruction | Collisions | Final coarse collapse | Usage-window failures | Mean oracle TV | Mean oracle KL, nats |
|---|---:|---:|---:|---:|---:|---:|
| Minimal EMA | 100% | 0/20 | 6/20 | 7/20 | 0.268484 | 0.371336 |
| EMA + recovery | 99.9533% | 0/20 | 0/20 | 0/20 | 0.122937 | 0.109387 |
| Reference | 99.9533% | 0/20 | 0/20 | 0/20 | 0.121543 | 0.105068 |
| Fixed binary | 100% | 0/20 | 0/20 | 0/20 | 0.122163 | 0.106556 |

A usage-window failure means more than 5% collapsed minibatches in the final 200 observations. All models have 16 distinct encodings, complete factorized-oracle output support and zero outside-domain mass. The minimum sequence reconstruction for recovery and reference is 99.066%, in one model each; they still exceed the existing 98% criterion. Before-replay metrics are retained in results.json.

Recovery reduces mean TV by 0.145547 relative to minimal EMA, largely through preventing collapse in three seeds. It is 0.001395 worse than reference on average. Fixed binary is 0.000774 better than recovery, with seed-averaged differences ranging from -0.010123 to +0.006007. These small latter differences do not establish superiority or noninferiority. Treat the five seeds as replication units; objective arms sharing a seed are not 20 independent draws.

The second study repeated recovery exactly: all oracle TV values match the first study. Reference results also agree with the earlier frozen-codec analysis to numerical precision.

## Timing and operation estimates

Initial training-window timings are preserved per model, with compilation, shared initialization, data preparation and replay separately recorded. Some initial windows overlapped with the test suite; use the separate clean timing study for comparisons.

Clean timing repeats the identical seed-13 400-update trajectory for each objective arm three times, with rotated treatment order and resident inputs. Compilation and discarded warmup are excluded; execution is synchronized at each trajectory boundary. No other task-dispatched experiment or test ran concurrently. System background activity is uncontrolled.

| Condition | Median seconds / 400 updates | Observed range, seconds | Compiler FLOPs / update |
|---|---:|---:|---:|
| Minimal EMA | 0.25925 | 0.25553–0.32629 | 10,525,149 |
| EMA + recovery | 0.26759 | 0.26479–0.27863 | 10,564,359 |
| Reference | 0.26834 | 0.26363–0.28341 | 10,564,359 |
| Fixed binary | 0.25788 | 0.25553–0.26590 | 10,525,149 |

Fixed binary's median is about 3.6% lower than recovery's in this small CPU measurement; ranges overlap and the operation estimate differs by only 0.37%. This is not evidence of substantial acceleration, TPU savings, or end-to-end generation speedup. Recovery and reference have identical estimated FLOPs; the stateless centroid step's scalar value does not change graph complexity.

## Gate correction and provenance

The frozen execution protocols and original summary files used zero joint-oracle TV as a conservative support check. That condition is stronger than full factorized-oracle output support and flags one reconstruction-imperfect model in recovery/reference despite preserved support. The original files are preserved unmodified. The supplementary audit explicitly measures missing teacher mass under the factorized oracle; analysis.json and this report use that measurement, the existing 98% reconstruction threshold, collision checks, perplexity >=1.5, and <=5% collapsed-window threshold. No model selection or training was redone after this correction. The maintained runner now uses the direct support measure.

Protocols were saved before training, with complete package source hashes; source snapshots and checkpoints remain in runs/maintenance-ablation and runs/binary-maintenance-comparison. The second protocol explicitly records selection of recovery as the simplest stable candidate after observing the first study. Hashes, per-model timings, metrics and supplementary audits are archived alongside this report. Results for all conditions preserve the shared entropy and schedule mechanisms: success does not show these are unnecessary, nor that data-driven initialization is unnecessary.

## Recommendation

Keep the existing production default. Prefer recovery over unguarded EMA for further learned-codebook experiments. Test fixed coarse binary endpoints on newly reserved seeds and additional data distributions as a simpler candidate; its present success suggests ongoing coarse-centroid learning is unnecessary in this particular experiment. It does not establish that fine codebooks, larger vocabularies or FSQ can be simplified similarly. Code-factorization error remains around 0.12 TV under every stable condition and is still the larger research issue.

## Validation and reproduction

The existing suite passed 44 tests. Two additional tests passed, checking recovery versus ordinary EMA and fixed-coarse/fine-EMA behavior. All 100 supplementary support evaluations reproduce original oracle TV within 1e-9; conditional-TC identities and law normalization are asserted during evaluation. git diff --check passed.

From a fresh output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.maintenance_experiment
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.maintenance_experiment --binary
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python examples/audit_maintenance.py
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python examples/maintenance_timing.py
```

The runner refuses existing output directories, preventing accidental overwrite. The runner reads the committed compact teacher-domain input in this folder and the historical report under reports/coarse-codebook-stability; those inputs are supplied by a clean checkout. Neither a new external dataset nor a full FSQ experiment was performed.
