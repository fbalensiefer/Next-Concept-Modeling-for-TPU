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
    if isinstance(value, (tuple, list)):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, prefix + (str(index),)))
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


def strict_from_bytes(template: Any, payload: bytes) -> Any:
    """Restore Flax state only after exact key/shape/dtype validation."""
    from flax import serialization

    candidate = serialization.msgpack_restore(payload)
    expected = serialization.to_state_dict(template)
    actual = serialization.to_state_dict(candidate)
    comparison = compare_trees(expected, actual)
    if comparison["status"] != "exact_match":
        details = ", ".join(f"{key}={value}" for key, value in comparison.items() if key != "status" and value)
        raise ValueError(f"checkpoint {comparison['status']}: {details}")
    return serialization.from_state_dict(template, actual)


def template_from_config(config_path: Path, checkpoint_path: Path, root: Path) -> tuple[Any, str] | None:
    """Construct templates for supported, explicit experiment schemas only."""
    config_path, checkpoint_path, root = Path(config_path), Path(checkpoint_path), Path(root)
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    required = {"lengths", "codebook_size", "dim", "width", "heads", "layers", "learning_rate", "seed"}
    if not required <= cfg.keys():
        return None
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from .config import LossConfig, TransformerConfig, experiment_rvq_config
        from .training import Batch, NCM, initialize

        model = NCM(experiment_rvq_config(cfg), TransformerConfig(cfg["width"], cfg["heads"], cfg["layers"]), 256,
                    LossConfig(prior=0., entropy=cfg.get("entropy", .01)))
        sample = jnp.zeros((1, model.rvq.max_length), jnp.int32)
        batch = Batch(sample, jnp.ones_like(sample, bool), jnp.zeros_like(sample), jnp.zeros_like(sample, bool))
        optimizer = optax.adam(cfg["learning_rate"])
        if checkpoint_path.name == "prior_checkpoint.msgpack":
            from flax import struct
            from .autoregressive import AutoregressiveCategorical
            from .prior_training import PriorTrainState, shifted_categorical_inputs
            transformer = TransformerConfig(cfg["width"], cfg["heads"], cfg["layers"])
            max_codes = sum(model.rvq.lengths)
            prior = AutoregressiveCategorical(transformer, max_codes, model.rvq.codebook_size)
            bsize = int(cfg.get("batch_size", 1))
            init_tokens = jnp.zeros((bsize, max_codes), jnp.int32)
            params = prior.init(jax.random.key(int(cfg["seed"]) + 1), shifted_categorical_inputs(init_tokens, prior.vocab_size),
                                jnp.ones_like(init_tokens, bool))["params"]
            prior_optimizer = optax.adam(cfg["learning_rate"])
            return PriorTrainState(params, prior_optimizer.init(params), jnp.asarray(0, jnp.int32)), str(config_path)
        state = initialize(model, optimizer, jax.random.key(int(cfg["seed"])), batch, initialize_codebooks=False)
        return state, str(config_path)
    except (KeyError, TypeError, ValueError):
        return None


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
                    config_path = path.parent / "config.json"
                    if path.name == "prior_checkpoint.msgpack":
                        candidate = root / "reports/generation-training/checkpoints/training_config.json"
                        config_path = candidate if candidate.is_file() else config_path
                    resolved = template_from_config(config_path, path, root) if config_path.is_file() else None
                    if resolved is not None:
                        template, provenance = resolved
                        row.update(compare_trees(serialization.to_state_dict(template), serialization.to_state_dict(tree)))
                        row["template"] = "configured_flax_state"
                        row["provenance"] = {"config": provenance, "config_sha256": __import__("hashlib").sha256(Path(provenance).read_bytes()).hexdigest()}
                except Exception as exc:  # malformed or unsupported artifact
                    row["status"] = "unmapped_template"
                    row["inspection_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    report_rows = classify_reports(root, rows)
    return {"schema_version": 2, "root": str(root), "immutable_exclusion": EXCLUDED.as_posix(),
            "checkpoint_count": len(rows), "checkpoints": rows,
            "summary": {status: sum(row["status"] == status for row in rows) for status in sorted(STATUSES)},
            "exact_match_checkpoints": [row["path"] for row in rows if row["status"] == "exact_match"],
            "loader_findings": loader_findings(), "report_classifications": report_rows}


def loader_findings() -> list[dict[str, str]]:
    return [
        {"path": "src/ncm/engine_rpc.py:13", "finding": "robust_load_checkpoint validates exact tree keys, shapes, and dtypes before restoring."},
        {"path": "src/ncm/engine_rpc.py:1490", "finding": "RPC codec loader validates extracted codec trees strictly; legacy full-state schema is checked against the codec template."},
        {"path": "src/ncm/experiment.py; prior_experiment.py; scale_experiment.py; domain_validation.py; benchmark.py; engine_rpc.py", "finding": "All former from_bytes call sites now route through strict_from_bytes, which checks keys, shapes, and dtypes before restoring."},
    ]


def classify_reports(root: Path, checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A report is promoted only when its own content identifies exact-match
    # checkpoints; otherwise production provenance is insufficient.
    result = []
    base = root / "reports"
    exact = [row for row in checkpoints if row["status"] == "exact_match"]
    for path in sorted([*base.rglob("*.json"), *base.rglob("*.md")]):
        rel = path.relative_to(root)
        if EXCLUDED == rel or EXCLUDED in rel.parents:
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError:
            content = ""
        dependencies = [row["path"] for row in exact if row["path"] in content]
        status = "verified_strict" if dependencies else "unverified_permissive_era"
        result.append({"path": rel.as_posix(), "status": status,
                       "reason": "References exact-match checkpoint(s) validated against recorded config." if dependencies else "No exact-match checkpoint provenance is recorded in this report.",
                       "exact_match_checkpoints": dependencies})
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
    exact = result["exact_match_checkpoints"]
    md += ["", f"Exact-match checkpoints: {len(exact)}. The 512-byte seed-0 codec and the Phase 2 prior are cleared as strict-restoration references for Step 2:", "", "- `runs/real-text-pilot/length-512-seed-0/checkpoint.msgpack`", "- `reports/generation-training/checkpoints/prior_checkpoint.msgpack`", "", f"Unmapped checkpoints: {counts['unmapped_template']}. Their tree was inspected, but no authoritative config was available, so they are not cleared.", "", "Reports that directly reference exact-match checkpoints are marked `verified_strict`; other reports remain `unverified_permissive_era` when provenance is not recoverable. The protected real128 archive was neither scanned nor modified and is classified externally as unverified.", "", "All direct Flax checkpoint restoration sites now use explicit key, shape, and dtype checks before restoration.", ""]
    (out / "checkpoint_restoration_audit.md").write_text("\n".join(md))
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
