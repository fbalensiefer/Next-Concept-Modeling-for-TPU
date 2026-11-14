# TC probe decision

git_commit: ca7308cccd50fbc54346b53d1e7463ee4eb1ef01

Numerical gate pass fraction: 0.773148; required 0.90.
Strict disjoint-sequence-identity gate: False.
Real-text gate: False.

Synthetic pooled estimated TC share: coarse 0.823395, fine 0.176605.
Synthetic >=30% coarse-share decision: True.
These pooled finite-model estimates are biased diagnostics, not real-text TC.

No TC claim is permitted on real text. Real-text code probing is stopped.
Copied-bit has only two distinct sequences; identity-disjoint train/eval
with the same nondegenerate source law is impossible. Overlaps and numerical
failures remain in the table. No additional hybrid models are trained.

Reporting clarification (git_commit: a1ed801): no model is added in response
to this failed probe. The independently preregistered benchmark hybrid arms
proceed as planned; their training is not a TC-probe decision.
