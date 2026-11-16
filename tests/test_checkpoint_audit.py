import numpy as np

from ncm.checkpoint_audit import compare_trees, scan, strict_from_bytes, template_from_config


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


def test_strict_flax_restore_rejects_dtype_mismatch():
    from flax import serialization

    template = {"params": {"w": np.zeros((2, 3), np.float32)}}
    bad = serialization.to_bytes({"params": {"w": np.zeros((2, 3), np.float64)}})
    try:
        strict_from_bytes(template, bad)
    except ValueError as error:
        assert "dtype_mismatch" in str(error)
    else:
        raise AssertionError("dtype mismatch was silently accepted")


def test_config_reconstructs_authoritative_ncm_template(tmp_path):
    import jax
    import jax.numpy as jnp
    import optax
    from ncm.config import LossConfig, TransformerConfig, experiment_rvq_config
    from ncm.training import Batch, NCM, initialize
    from flax import serialization

    cfg = {"lengths": [2, 4], "codebook_size": 4, "dim": 4, "width": 8,
           "heads": 2, "layers": 1, "learning_rate": .001, "seed": 3}
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(cfg))
    model = NCM(experiment_rvq_config(cfg), TransformerConfig(8, 2, 1), 256, LossConfig(prior=0.))
    tokens = jnp.zeros((1, 4), jnp.int32)
    batch = Batch(tokens, jnp.ones_like(tokens, bool), jnp.zeros_like(tokens), jnp.zeros_like(tokens, bool))
    state = initialize(model, optax.adam(.001), jax.random.key(3), batch, initialize_codebooks=False)
    checkpoint_path = tmp_path / "checkpoint.msgpack"
    resolved = template_from_config(config_path, checkpoint_path, tmp_path)
    assert resolved is not None and resolved[1] == str(config_path)
    assert compare_trees(serialization.to_state_dict(resolved[0]), serialization.to_state_dict(state))["status"] == "exact_match"

