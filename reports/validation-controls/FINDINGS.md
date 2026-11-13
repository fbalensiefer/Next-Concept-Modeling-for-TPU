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
