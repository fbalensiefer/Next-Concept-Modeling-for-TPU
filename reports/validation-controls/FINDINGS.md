# Validation findings

git_commit: 94efd3f (test-only validation revision).

## Proven bug: single-level hybrid initialization

The supported exact-bucket test failed before cache implementation: Flax Embed
with one embedding and a scalar index raised a broadcast error. Indexing with
an explicit one-element array and selecting its first element fixes the
single-level case and preserves the multi-level computation. The existing
cross-bucket regression test is retained. This fix is committed separately.

Further study outcomes will be appended without deleting failures.

## Smoke attempt 1: report serialization failure

git_commit: 2c9fd78. Seed 200 completed all smoke timing cases, but the final
JSON writer rejected a NumPy boolean teacher-warning statistic. This is an
infrastructure failure, not an altered test threshold or a new seed. The writer
now converts NumPy scalars/arrays to JSON values, with a regression test. The
same seed and protocol are rerun after the fix; the failed attempt remains here.

## Discarded timing attempt: competing CPU validation jobs

git_commit: 09eac68. The first real128 timing attempt overlapped the tail of
the regression suite and a further contract check. Its partial results are
retained under `real128-contended/` and do not support latency comparisons.
The run was interrupted, without replacing any seed or altering a threshold.
The complete frozen grid is rerun with the same seeds after tests finish and
without other agent-started compute jobs. This records an execution/fairness
failure rather than silently keeping favorable timing observations.

## Zero-prompt architecture fairness

git_commit: 95ce7e5 (before clean real128). The auto(0,0) public generator selects
the historical padded implementation. A naive zero-prompt parallel comparison
would therefore change attention primitives and grid execution as well as AR
factorization. The control runner now instantiates the existing HybridGenerator
core for both arms and uses empty prompt embeddings at P=0 to avoid unused prompt
lookup weights in parameter matching. Existing public defaults remain unchanged.
The discarded attempt's P=0 quality contrast is not interpreted as an AR-only
effect. This fixes experiment construction, not a statistical threshold.

## Frozen checks completed; model sweep in progress

git_commit: a1ed8019578ea1fd77f45b871cdd55a1b5157af2. All 118 tests pass on four virtual CPU devices.
Power TV is 0.00675750 against a fixed 0.00826506 floor; all four negative
controls are rejected. The learned TC gate passes 77.3148% of numerical cases,
below 90%; real-text TC probing is stopped. Training-size pass rates are
63.8889%, 80.5556%, and 87.5%. The uniform zero-TC control fails 9/36 cases
and noisy parity fails 19/36. No thresholds or seeds were changed.
The complete 351-row real128 timing/reference sweep remains in progress.

The first clean seed-200/P=0 teacher has exact held-out BPB 8.4623, worse
than the 8-BPB uniform-byte law. This is retained as a negative result; no
training budget, seed, or quality threshold is adjusted in response. It further
limits interpretation to pipeline feasibility and finite exactness checks.

## Parameter-matching audit

git_commit: a1ed801. The fixed heads=2, layers=1 AR width grid cannot attain
the exact NCM parameter counts. Actual seed-200 counts are AR6920/NCM6040
(P=0, +14.5695%), AR8826/NCM8224 (P=16, +7.3200%), and AR8826/NCM8608
(P=64, +2.5325%). Counts depend on shapes, not seed. These mismatches are
reported explicitly; no architecture is retuned after seeing results. The
comparison is not parameter-perfect and cannot establish a general NCM
student advantage. Numerical quality matching is independently unproven.

## Final audit: legacy single-level initialization

git_commit: a1ed801 (measurement revision). The same Flax scalar-index bug
also affected the legacy scanned generator when K=1. A new regression failed
with the original broadcast error; a one-element index fixes it without
changing multi-level parameter shapes. This is committed separately.
The registered three-level experiment is unaffected.
