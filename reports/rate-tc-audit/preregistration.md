# Rate–TC audit preregistration

**Protocol:** `rate-tc-audit-stage-a-v1`. This protocol is frozen before Stage A. Stage A may read synthetic generators and analytical laws only. Real text and checkpoints are prohibited.

## Estimand

For each target coordinate, the fixed local context `phi_i(Z_<k)` consists of categorical codes at nearest aligned parent locations plus a deterministic summary bin computed only from those parent codes. The independent model is a smoothed categorical table `q_ind(z_i | phi_i)`. A maximum spanning tree is selected on fit data using conditional pairwise MI pooled within context strata. Its root conditions on `phi_i`; each other node conditions on `phi_i` and the selected parent code. A bounded causal AR diagnostic conditions on `phi_i` and the two immediately preceding within-level codes. Every table uses symmetric Dirichlet alpha 0.5 per outcome category; tree selection uses fit data only.

The primary finite-sample statistic is held-out per-sequence log-loss improvement `DeltaH_tree = Hhat_eval(q_ind) - Hhat_eval(q_tree)`. Evaluation sequences are independent of fitting and tree selection; intervals resample whole sequences. Conditional on the fitted model, evaluation cross entropy is unbiased for `H(P,q)=H(P)+KL(P||q)`, which upper-bounds entropy. The difference estimates out-of-sample pairwise-tree predictive gain. It does not estimate true TC without bias and is not a finite-sample TC lower bound. Bootstrap intervals quantify evaluation-sample variability conditional on the fit; they do not include model-selection uncertainty.

## Stage A gates

- Independent uniform: 4,096 sequences; evaluation sizes 66 and 512; tree gain must be at most 0.02 nats per sequence.
- Rich-support disjoint identity: 2,048 fit and 66 evaluation sequences; zero full-identity overlap, conditional independence given local context, and gain at most 0.02 nats per sequence.
- Conditional Markov and block-correlated laws: 100 independent replications at evaluation sizes 66 and 512. At least 95% of estimates must be within 20% relative error of analytical dependence; nominal 95% sequence-bootstrap interval coverage must be at least 90%.
- Three-bit parity: pairwise-tree gain magnitude at most 0.02 nats per sequence and AR gain at least 0.10 nats per sequence, documenting the tree’s higher-order blind spot.

Any failed gate halts the audit. Stage B remains unauthorized until every Stage A gate passes. Its preregistered kill rule rejects pure parallel K=3 when the lower 95% bootstrap endpoint for finest-level predictive dependence exceeds 128 nats per sequence, or exceeds the total measured coarse-level marginal-rate reduction.
