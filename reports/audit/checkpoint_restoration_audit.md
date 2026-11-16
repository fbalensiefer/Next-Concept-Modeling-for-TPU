# Checkpoint restoration audit

Scanned 1535 msgpack files under runs/ and reports/, excluding the immutable real128 archive.

## Checkpoint statuses

- `dtype_mismatch`: 0
- `exact_match`: 17
- `extra_leaves`: 0
- `file_missing`: 0
- `missing_leaves`: 0
- `shape_mismatch`: 0
- `unmapped_template`: 1518

Exact-match checkpoints: 17. The 512-byte seed-0 codec and the Phase 2 prior are cleared as strict-restoration references for Step 2:

- `runs/real-text-pilot/length-512-seed-0/checkpoint.msgpack`
- `reports/generation-training/checkpoints/prior_checkpoint.msgpack`

Unmapped checkpoints: 1518. Their tree was inspected, but no authoritative config was available, so they are not cleared.

Reports that directly reference exact-match checkpoints are marked `verified_strict`; other reports remain `unverified_permissive_era` when provenance is not recoverable. The protected real128 archive was neither scanned nor modified and is classified externally as unverified.

All direct Flax checkpoint restoration sites now use explicit key, shape, and dtype checks before restoration.
