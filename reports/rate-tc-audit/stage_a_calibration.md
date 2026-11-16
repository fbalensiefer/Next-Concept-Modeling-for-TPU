# Stage A replicated synthetic calibration

**Verdict: HALT. Stage B real-text evaluation was not run.** The 100-replication sweep used only generated synthetic sources and analytical law parameters; every cell used 2,000 sequence bootstrap resamples. The preregistered requirement is at least 95% relative-error passes within 20% and at least 90% empirical coverage of nominal 95% intervals.

| Law | Eval N | Mean gain | Bias | Accuracy pass rate | 95% interval coverage | Gate |
|---|---:|---:|---:|---:|---:|---|
| Conditional Markov | 66 | 1.1104 | +0.0062 | 90% | 97% | FAIL |
| Conditional Markov | 512 | 1.1056 | +0.0014 | 100% | 93% | PASS |
| Pair-correlated blocks | 66 | 0.5264 | -0.0144 | 65% | 93% | FAIL |
| Pair-correlated blocks | 512 | 0.5416 | +0.0007 | 100% | 97% | PASS |

TC and gains are in nats per four-coordinate sequence. Both laws meet the bootstrap coverage threshold at both sizes. The accuracy gate fails at N=66 for both laws, with larger failure for the block source. Because every calibration cell must pass, the estimator is not cleared for the 66-window real-text setting. Uniform, rich-support disjoint-identity and parity diagnostics passed; see the JSON for their values and all per-replication gains and intervals.

No real-text splits or checkpoint files were opened. The out-of-sample gain remains a predictive comparison between fitted models, not a finite-sample lower bound on true TC.
