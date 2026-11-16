"""Read-only inventory and template comparison for Flax msgpack checkpoints.

Run with ``python -m ncm.checkpoint_audit --root .``. The scanner never writes
under runs/ or reports/; its JSON/Markdown outputs are written only to
reports/audit. Checkpoints lacking a uniquely recoverable model template are
reported as unmapped_template rather than guessed from tensor shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STATUSES = {"exact_match", "missing_leaves", "extra_leaves", "shape_mismatch", "dtype_mismatch", "file_missing", "unmapped_template"}
EXCLUDED = Path("reports/validation-controls/real128")


def _flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + (str(key),)))
        return result
    return {prefix: value}


def compare_trees(template: Any, checkpoint: Any) -> dict[str, Any]:
    """Compare complete tree keys, array shapes and dtypes without restoring."""
    expected, actual = _flatten(template), _flatten(checkpoint)
    missing = sorted("/".join(k) for k in expected.keys() - actual.keys())
    extra = sorted("/".join(k) for k in actual.keys() - expected.keys())
    shape, dtype = [], []
    for key in expected.keys() & actual.keys():
        a, b = expected[key], actual[key]
        if hasattr(a, "shape") or hasattr(b, "shape"):
            if tuple(getattr(a, "shape", ())) != tuple(getattr(b, "shape", ())):
                shape.append("/".join(key))
            elif str(getattr(a, "dtype", type(a).__name__)) != str(getattr(b, "dtype", type(b).__name__)):
                dtype.append("/".join(key))
    status = ("missing_leaves" if missing else "extra_leaves" if extra else
              "shape_mismatch" if shape else "dtype_mismatch" if dtype else "exact_match")
    return {"status": status, "missing_leaves": missing, "extra_leaves": extra,
            "shape_mismatch": shape, "dtype_mismatch": dtype}


def scan(root: Path) -> dict[str, Any]:
    """Inventory every msgpack file, excluding the immutable real128 tree."""
    try:
        from flax import serialization
    except ImportError as exc:
        raise RuntimeError("Flax is required to inspect msgpack checkpoints") from exc
    root = root.resolve()
    rows = []
    for base in (root / "runs", root / "reports"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.msgpack")):
            rel = path.relative_to(root)
            if rel == EXCLUDED or EXCLUDED in rel.parents:
                continue
            row = {"path": rel.as_posix(), "status": "unmapped_template",
                   "template": None, "provenance": None, "dependent_reports": []}
            if not path.is_file():
                row["status"] = "file_missing"
            else:
                try:
                    tree = serialization.msgpack_restore(path.read_bytes())
                    row["leaf_count"] = len(_flatten(tree))
                    row["tree_keys"] = sorted("/".join(k) for k in _flatten(tree))
                    row["sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                except Exception as exc:  # malformed or unsupported artifact
                    row["status"] = "unmapped_template"
                    row["inspection_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return {"schema_version": 1, "root": str(root), "immutable_exclusion": EXCLUDED.as_posix(),
            "checkpoint_count": len(rows), "checkpoints": rows,
            "summary": {status: sum(row["status"] == status for row in rows) for status in sorted(STATUSES)},
            "loader_findings": loader_findings(), "report_classifications": classify_reports(root)}


def loader_findings() -> list[dict[str, str]]:
    return [
        {"path": "src/ncm/engine_rpc.py:13", "finding": "Current robust_load_checkpoint performs strict key/shape restoration; does not explicitly compare dtype."},
        {"path": "src/ncm/engine_rpc.py:1490", "finding": "RPC codec loader validates extracted codec trees strictly; legacy full-state schema is checked against the codec template."},
        {"path": "src/ncm/experiment.py:78", "finding": "serialization.from_bytes uses a full train-state template; Flax validates structure and shape, dtype equality is not asserted."},
        {"path": "src/ncm/prior_experiment.py:31,69", "finding": "Direct Flax restoration against codec/prior templates; structure and shapes checked, dtype equality not asserted."},
        {"path": "src/ncm/scale_experiment.py:152,296,497,549", "finding": "Direct Flax restoration against teacher/repair/codec/prior templates; structure and shapes checked, dtype equality not asserted."},
        {"path": "src/ncm/domain_validation.py:215", "finding": "Direct Flax restoration against initialized codec state; structure and shapes checked, dtype equality not asserted."},
        {"path": "src/ncm/benchmark.py:116", "finding": "Direct Flax restoration against initialized AR state; structure and shapes checked, dtype equality not asserted."},
        {"path": "src/ncm/engine_rpc.py:1535,1540", "finding": "Direct Flax restoration for AR and NCM states; structure and shapes checked, dtype equality not asserted."},
    ]


def classify_reports(root: Path) -> list[dict[str, str]]:
    # Report artifact timestamps and commits are not consistently embedded.
    # Conservatively flag all reports whose production path is not proven strict.
    result = []
    base = root / "reports"
    for path in sorted([*base.rglob("*.json"), *base.rglob("*.md")]):
        rel = path.relative_to(root)
        if EXCLUDED == rel or EXCLUDED in rel.parents:
            continue
        result.append({"path": rel.as_posix(), "status": "unverified_permissive_era",
                       "reason": "Production revision/loader provenance is not established from the report artifact alone."})
    result.append({"path": EXCLUDED.as_posix(), "status": "unverified_permissive_era",
                   "reason": "Protected archive excluded from file reads and writes; status recorded externally."})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = scan(args.root)
    out = args.root.resolve() / "reports/audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint_restoration_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    counts = result["summary"]
    md = ["# Checkpoint restoration audit", "", f"Scanned {result['checkpoint_count']} msgpack files under runs/ and reports/, excluding the immutable real128 archive.", "", "## Checkpoint statuses", ""]
    md.extend(f"- `{key}`: {value}" for key, value in counts.items())
    md += ["", "Every checkpoint is currently `unmapped_template`: artifacts were inventoried read-only, but the scanner does not infer templates from shapes or nearby names. Consequently this audit is not a strict-restoration clearance for the bounded benchmark.", "", "The report classifications are conservative: without per-run production commits, reports are marked `unverified_permissive_era`. The real128 archive was excluded from scanning and is classified externally.", "", "Direct Flax `serialization.from_bytes` checks tree structure and array shapes against its template, but does not establish exact dtype equality. See the machine-readable loader findings for call sites.", ""]
    (out / "checkpoint_restoration_audit.md").write_text("\n".join(md))
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
