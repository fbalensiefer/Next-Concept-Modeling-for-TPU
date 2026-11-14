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

## Final audit: analytic operation model corrections

git_commit: a1ed801 (original estimates). Compiler regression tests confirm
cache-only prefill in a one-layer AR has no floating-point attention cost
dependent on cache capacity: the last layer's attention/MLP/head is discarded.
The analytic model incorrectly charged that work. A second compiler check
confirms padded RVQ executes the same distance shapes when active lengths
change inside a fixed K/L bucket; the analytic distance count incorrectly
used active lengths. Both original tests failed, with their logs retained in
accounting-audit-before.txt. These estimation fixes change no model samples,
BPB values, acceptance counts, timings, thresholds, or seeds. Original a1ed801
estimates remain in the immutable experiment folders; final tables retain
raw-estimate columns and identify the corrected analytic revision.

<!-- final-study-summary -->
## Completed frozen studies

git_commit: bbf82c63d738633b950e5daef5ca862e55f39016; model measurement commits are retained in every table row.

- Powered exactness: TV 0.00675750, frozen floor 0.00826506; all four corrupt verifier controls rejected.
- Coarse cache: sampled codes match recomputation for seeds 200–202; logits pass 1e-5, including two coarse levels and masked prompts/grids. AR padding cache is tested at every position.
- TC gate: 77.31% numerical pass rate versus 90%; strict disjoint identity gate also failed. No real-text TC claim is permitted. The protocol and all 216 cases are retained. Numerical sensitivity to sample size is in tc_probe_sensitivity.csv.
- Real text: three seeds, 128-byte continuations, 32 fixed windows per split including the final eligible partial window. All prompt buckets share continuation hashes and masks. 351 cached-reference/drafter rows, 45 quality rows, and 135 checkpoint observations completed.
- Distilled AR drafter is present for every draft length, prompt length, batch and seed. NCM has lower raw median latency in 36/108 paired cases. This is a bounded weak-model CPU pilot, not a general drafter advantage claim.
- Teacher BPB is within 0.3 of the small AR in 3/9 seed/context cases. All teacher sizes, steps and BPB are disclosed; eight steps do not establish a strong teacher. The teacher is worse than the 8-BPB uniform-byte reference in 7/9 cases.
- AR width matching is approximate on the fixed heads=2/layers=1 grid. Actual mismatch fractions are retained in bpb_vs_train_flops.csv; the largest is about 14.6%, so these are not parameter-perfect student comparisons.
- All NCM student BPB values are upper bounds. Intersections of these bounds with exact AR likelihood curves do not establish equal quality. Non-overlap is reported without extrapolation.
- Verified drafters have lower raw median latency than the cached teacher in 0/324 cases under this full-prefix verifier. This is a local pilot result, not a general performance law.
- Exact-teacher drafter comparisons are explicitly matched by distribution in latency_at_matched_bpb.csv. Per-arm student checkpoint curves and exploratory three-seed paired bootstrap intervals are separate.
- Final analytic estimates correct the compiler-validated prefill and padded-distance accounting bugs. Raw a1ed801 estimates remain in the immutable study directory and raw_* columns; analytic_revision.json records the corrections. No measured timing, BPB, acceptance or round count was altered.
- Compiler FLOPs are compiler estimates. Analytic FLOPs are operation-model estimates. Speculative estimates use all thirty timed round-count observations, including masked finished lanes in batched loops. Prefill and teacher/codec prerequisites, teacher distillation forwards, and 32 teacher-sample generations are charged without double-counting those forwards.
- Initial prompt bytes are identical across drafters. During later rounds, NCM uses its most recent fixed prompt bucket, while the AR drafter uses the full committed prefix. These context policies are explicit in the table and limit general architecture conclusions.
- Teacher verification remains full-prefix. NCM produces its complete 128-token latent block even when only 2/4/8/16 decoded token probabilities are used. Both costs are measured. Cached coarse levels remove recomputation but do not eliminate these costs.
- No TPU evidence or general compute-saving claim exists. The discarded contended timing attempt and serialization failure remain recorded above.

## Final test audit

git_commit: bbf82c63d738633b950e5daef5ca862e55f39016. All **121 tests passed** in 232.66 seconds on four virtual CPU devices, including all 81 prior tests and 40 added tests. The final full-suite log is in tests-final.txt; tests.json records source hashes. Earlier test evidence is retained in tests-before-final-audit.json. The three compiler/initialization regressions passed after their separately committed fixes.
