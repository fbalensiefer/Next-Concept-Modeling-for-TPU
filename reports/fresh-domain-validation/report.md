# Fresh-seed, broader-distribution validation

Completed 360 paired model runs on six prespecified full-support binary source distributions. The fixed binary coarse quantizer does not validate as a general replacement. All three conditions encounter failures outside the original domain, and factorized generation can remain poor despite perfect reconstruction. Production defaults were not changed.

## Protocol and scope

Student seeds 31–35 were not present in the earlier archived experiments. Six source laws, two sequence lengths, all four objective arms, and three stable quantizers were fixed before training. Each model received 400 codec updates and 200 decoder-only replay updates. Six independently initialized width-16 causal feature teachers received 400 updates each. Data budgets remained 512 training, 512 validation and 256 calibration draws; examples can repeat across splits, and this is not unseen-support testing.

The paired treatments share data, parameters, K-means initialization, minibatch order, neural architecture, objectives, regularizers and schedules. Initial-state hashes match within every domain/seed across all arms and treatments. Fixed binary means fixed two-vector coarse endpoints with a trainable encoder and fine-book EMA; it is not full FSQ. No learned prior was trained: generation metrics are exact population-law oracles for the frozen codec.

The source law itself is the sampling/evaluation target. Feature teachers supply only PCA alignment targets, preventing teacher fitting limitations from replacing the intended source dependence. This differs from the older pilot, which targeted a fitted teacher law; do not read changes from old TV values as a causal improvement. The original causal feature-alignment convention is intentionally retained to isolate maintenance.

| Source | Length | Definition | Distinct training sequences | Source-to-feature-teacher KL |
|---|---:|---|---:|---:|
| original_mixture | 4 | 0.75 constant modes + 0.25 uniform | 16 | 0.02801 |
| biased_independent | 4 | Independent Bernoulli(0.2) symbols | 15 | 0.00263 |
| persistent_markov | 8 | Fair initial bit; 0.85 probability of staying | 80 | 0.01067 |
| noisy_parity | 8 | 0.90 uniform even parity + 0.10 uniform | 148 | 0.49557 |
| skewed_modes | 8 | 0.80 all A + 0.10 all B + 0.10 uniform | 47 | 0.00893 |
| alternating_modes | 8 | 0.80 alternating modes + 0.20 uniform | 81 | 0.03201 |

## All results after matched decoder replay

Fidelity failure means any prespecified criterion fails: sequence reconstruction >=98%, missing source mass <=1%, outside-domain mass <=1%, or the collision-based reconstruction ceiling >=98%. All are population-weighted. Coarse balance is diagnostic rather than a universal criterion: a skewed source need not require balanced coarse codes. An encoding collision is also reported even when its probability mass is below the failure tolerance.

| Source | Condition | Mean reconstruction | Fidelity failures /20 | Models with any collision | Mean oracle TV | Mean oracle KL, nats | Final coarse collapse /20 |
|---|---|---:|---:|---:|---:|---:|---:|
| original_mixture | EMA + recovery | 97.891% | 2 | 2 | 0.14310 | 0.12644 | 0 |
| original_mixture | Reference | 99.766% | 1 | 1 | 0.12733 | 0.13423 | 0 |
| original_mixture | Fixed binary | 100.000% | 0 | 0 | 0.11858 | 0.11084 | 0 |
| biased_independent | EMA + recovery | 99.824% | 1 | 2 | 0.08741 | 0.13994 | 0 |
| biased_independent | Reference | 99.184% | 1 | 2 | 0.09256 | 0.14562 | 0 |
| biased_independent | Fixed binary | 100.000% | 0 | 0 | 0.08894 | 0.15095 | 0 |
| persistent_markov | EMA + recovery | 97.234% | 4 | 4 | 0.48523 | 0.97307 | 0 |
| persistent_markov | Reference | 96.240% | 6 | 6 | 0.48901 | 0.95434 | 0 |
| persistent_markov | Fixed binary | 100.000% | 0 | 0 | 0.49101 | 0.98283 | 1 |
| noisy_parity | EMA + recovery | 93.988% | 4 | 5 | 0.46989 | 0.73921 | 0 |
| noisy_parity | Reference | 95.416% | 4 | 6 | 0.47016 | 0.73217 | 0 |
| noisy_parity | Fixed binary | 87.605% | 4 | 5 | 0.49310 | 0.73359 | 0 |
| skewed_modes | EMA + recovery | 99.598% | 2 | 5 | 0.18229 | 0.34106 | 0 |
| skewed_modes | Reference | 99.445% | 2 | 6 | 0.18136 | 0.34177 | 0 |
| skewed_modes | Fixed binary | 99.520% | 3 | 4 | 0.17823 | 0.33114 | 0 |
| alternating_modes | EMA + recovery | 97.023% | 3 | 5 | 0.70474 | 2.80647 | 0 |
| alternating_modes | Reference | 96.809% | 3 | 6 | 0.74007 | 3.02846 | 0 |
| alternating_modes | Fixed binary | 94.352% | 4 | 5 | 0.70588 | 2.80654 | 1 |

## Failures by objective arm

R = reconstruction, A = feature alignment, D = dependence, AD = both. Counts are failed fidelity checks out of five seeds. These rows prevent an average over objective arms from hiding a problematic training objective.

| Source | Arm | EMA + recovery | Reference | Fixed binary |
|---|---|---:|---:|---:|
| original_mixture | D | 1 | 0 | 0 |
| original_mixture | AD | 1 | 1 | 0 |
| biased_independent | AD | 1 | 1 | 0 |
| persistent_markov | R | 1 | 1 | 0 |
| persistent_markov | A | 0 | 1 | 0 |
| persistent_markov | D | 1 | 2 | 0 |
| persistent_markov | AD | 2 | 2 | 0 |
| noisy_parity | A | 0 | 0 | 1 |
| noisy_parity | D | 2 | 2 | 1 |
| noisy_parity | AD | 2 | 2 | 2 |
| skewed_modes | D | 1 | 1 | 1 |
| skewed_modes | AD | 1 | 1 | 2 |
| alternating_modes | A | 0 | 0 | 1 |
| alternating_modes | D | 1 | 1 | 1 |
| alternating_modes | AD | 2 | 2 | 2 |

## Paired comparisons

Differences are candidate minus reference oracle TV, averaging objective arms within each seed first. Negative favors the candidate. Intervals enumerate all 5^5 bootstrap resamples of five seed means. They are descriptive, not multiplicity-adjusted noninferiority certificates. The prespecified 0.02 mean tolerance does not override fidelity failures.

| Source | Candidate | Mean TV difference | Descriptive 95% interval |
|---|---|---:|---|
| original_mixture | EMA + recovery | +0.01577 | [-0.00001, +0.04565] |
| original_mixture | Fixed binary | -0.00876 | [-0.02727, +0.00181] |
| biased_independent | EMA + recovery | -0.00516 | [-0.01877, +0.00269] |
| biased_independent | Fixed binary | -0.00362 | [-0.02913, +0.01630] |
| persistent_markov | EMA + recovery | -0.00378 | [-0.00919, +0.00030] |
| persistent_markov | Fixed binary | +0.00200 | [+0.00036, +0.00456] |
| noisy_parity | EMA + recovery | -0.00028 | [-0.00355, +0.00300] |
| noisy_parity | Fixed binary | +0.02293 | [-0.00522, +0.05114] |
| skewed_modes | EMA + recovery | +0.00094 | [-0.00479, +0.00650] |
| skewed_modes | Fixed binary | -0.00313 | [-0.01479, +0.00853] |
| alternating_modes | EMA + recovery | -0.03533 | [-0.07529, +0.00212] |
| alternating_modes | Fixed binary | -0.03419 | [-0.06856, -0.00250] |

## Mathematical controls and interpretation

The source-space controls use exact known conditionals, not learned networks, and are not matched-compute empirical baselines. They identify avoidable statistical restrictions:

- Biased independent symbols admit exact one-round generation without a codec. Nonzero learned-code oracle error here is introduced by the representation/factorization, not an intrinsic dependence of the data.
- The persistent Markov source admits exact midpoint-bridge sampling in four rounds: positions [3], [1,5], [0,2,4,6], [7], using zero-based positions. The selected points separate independent remaining intervals.
- Noisy parity admits exact two-round generation: seven independent fair bits, followed by the final bit conditioned on them. A schedule with one coarse bit followed by eight independent codes is a different restriction, despite also using two rounds.

Each construction was enumerated against the source law; residual TV is at floating-point roundoff. This demonstrates that dependence structure and the ordering of conditional decisions matter, rather than simply codebook balance or number of rounds. It does not establish that an unknown language distribution admits such simple exact schedules.

The clearest failure example is noisy-parity seed 31, AD: fixed binary reconstructs only 25% of source mass and misses 75% under its factorized oracle. Recovery and reference also show objective-dependent failures. Thus the earlier binary simplification succeeded in a limited envelope, and neither centroid learning nor freezing alone protects information against the shared weighted objectives. Keep fidelity and collision checks ahead of TC/utilization improvements. Nominal code capacity is sufficient: 32 codewords for 16 inputs or 512 for 256 inputs. Learned collisions are not forced by cardinality.

## Timing and resource record

Each result records synchronized codec/replay training windows, compiler-estimated step FLOPs, compilation, shared initialization and shared data preparation. Other task-dispatched tests were not run concurrently. Times exclude evaluation and checkpoint I/O. Background system load remains uncontrolled; these measurements do not establish a speed advantage or TPU economics. One interrupted codec window has no persisted timing and is recorded as null, not imputed.

| Length | Condition | Timed models | Median seconds /400 codec updates | Step FLOP estimates |
|---|---|---:|---:|---|
| 4 | EMA + recovery | 40/40 | 0.26552 | [10564359.0] |
| 4 | Reference | 40/40 | 0.26619 | [10564359.0] |
| 4 | Fixed binary | 40/40 | 0.25652 | [10525149.0] |
| 8 | EMA + recovery | 79/80 | 0.63091 | [24416086.0] |
| 8 | Reference | 80/80 | 0.63044 | [24416086.0] |
| 8 | Fixed binary | 80/80 | 0.63742 | [24349482.0] |

## Numerical incident and continuation

After 268 completed models, an evaluation assertion compared direct code KL with a float32 entropy identity and stopped. On the saved affected checkpoint, float32 entropy cancellation produced a 7.52394e-6 nat discrepancy; float64 entropy and direct float64 KL agree within 1.11e-15. The correction changes evaluation verification only. Training precision, loss, data, seeds and maintenance functions were unchanged. Function-AST checks preserve the maintenance, teacher preparation and source-law helpers.

The first 268 results were loaded unchanged, the affected model resumed from its completed trained checkpoint, and all remaining planned models were completed. Original protocol/source snapshots, continuation source versions, incident values and hashes are preserved. No failing seed was replaced and no result excluded.

## Recommendation and limits

Keep the current production default. Do not adopt fixed binary coarse endpoints as a generally validated replacement. Recovery remains a reasonable learned-codebook safeguard, but it does not solve the broader source/objective failures. The next experiment should protect reconstruction and collision fidelity while changing the dependency objective or sampling schedule; increasing maintenance complexity is not supported by these results.

This broadens synthetic dependency patterns, sequence length and feature teachers. It does not validate natural-language corpora, larger alphabets or codebooks, long contexts, cross-domain transfer, external pretrained teachers, learned-prior quality, or TPU performance. Each source has one feature teacher and five student seeds; statistical generality remains limited. Models train separately on each distribution, rather than transferring to an unseen domain.

## Artifacts and reproduction

protocol.json freezes the original design; results.json contains all 360 records; analysis.json includes domain, objective-arm and paired-seed summaries. teachers.json records feature-teacher fidelity. source_baselines.json contains exact source-space controls. numeric_incident.json and continuation.json document the evaluation correction. Ignored runs/fresh-domain-validation contains all model checkpoints, teacher-domain arrays, original source snapshots and continuation source versions.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.domain_validation
# If interrupted after protocol creation:
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python -m ncm.domain_validation --resume
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python examples/domain_source_baselines.py
PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python examples/summarize_domain_validation.py
```

The runner starts from the committed example configuration and generates its own analytic domains and feature teachers; unlike the preceding ablation, it does not require an ignored historical teacher checkpoint. The report-writing step is presentation only.
