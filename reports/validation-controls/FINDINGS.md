# Validation findings

git_commit: 94efd3f (test-only validation revision).

## Proven bug: single-level hybrid initialization

The supported exact-bucket test failed before cache implementation: Flax Embed
with one embedding and a scalar index raised a broadcast error. Indexing with
an explicit one-element array and selecting its first element fixes the
single-level case and preserves the multi-level computation. The existing
cross-bucket regression test is retained. This fix is committed separately.

Further study outcomes will be appended without deleting failures.
