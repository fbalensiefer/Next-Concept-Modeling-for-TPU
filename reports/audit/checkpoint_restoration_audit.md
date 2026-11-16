# Checkpoint restoration audit

Scanned 1535 msgpack files under runs/ and reports/, excluding the immutable real128 archive.

## Checkpoint statuses

- `dtype_mismatch`: 0
- `exact_match`: 0
- `extra_leaves`: 0
- `file_missing`: 0
- `missing_leaves`: 0
- `shape_mismatch`: 0
- `unmapped_template`: 1535

Every checkpoint is currently `unmapped_template`: artifacts were inventoried read-only, but the scanner does not infer templates from shapes or nearby names. Consequently this audit is not a strict-restoration clearance for the bounded benchmark.

The report classifications are conservative: without per-run production commits, reports are marked `unverified_permissive_era`. The real128 archive was excluded from scanning and is classified externally.

Direct Flax `serialization.from_bytes` checks tree structure and array shapes against its template, but does not establish exact dtype equality. See the machine-readable loader findings for call sites.
