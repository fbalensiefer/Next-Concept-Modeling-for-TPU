import numpy as np

from ncm.checkpoint_audit import compare_trees, scan


def test_compare_trees_detects_leaf_differences():
    template = {"params": {"w": np.zeros((2, 3), np.float32), "b": np.zeros((3,), np.float32)}}
    assert compare_trees(template, {"params": {"w": np.zeros((2, 3), np.float32), "b": np.zeros((3,), np.float32)}})["status"] == "exact_match"
    assert compare_trees(template, {"params": {"w": np.zeros((2, 3), np.float32)}})["status"] == "missing_leaves"
    assert compare_trees(template, {"params": {"w": np.zeros((2, 3), np.float32), "b": np.zeros((3,), np.float32), "extra": np.zeros(1)}})["status"] == "extra_leaves"
    assert compare_trees(template, {"params": {"w": np.zeros((3, 2), np.float32), "b": np.zeros((3,), np.float32)}})["status"] == "shape_mismatch"
    assert compare_trees(template, {"params": {"w": np.zeros((2, 3), np.float64), "b": np.zeros((3,), np.float32)}})["status"] == "dtype_mismatch"


def test_scan_is_read_only_and_excludes_real128(tmp_path):
    from flax import serialization

    run = tmp_path / "runs" / "example"
    run.mkdir(parents=True)
    checkpoint = run / "state.msgpack"
    checkpoint.write_bytes(serialization.to_bytes({"x": np.zeros((2,), np.float32)}))
    protected = tmp_path / "reports/validation-controls/real128/keep.msgpack"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"unchanged")
    before = checkpoint.read_bytes(), protected.read_bytes()
    result = scan(tmp_path)
    assert result["checkpoint_count"] == 1
    assert result["checkpoints"][0]["status"] == "unmapped_template"
    assert (checkpoint.read_bytes(), protected.read_bytes()) == before

