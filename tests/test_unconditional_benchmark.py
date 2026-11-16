import numpy as np
import jax.numpy as jnp

from ncm.unconditional_benchmark import (pareto_stop, reconstruction_metrics,
                                         timing_summary, xor_innovations, xor_inverse)


def test_xor_innovation_transform_is_lossless():
    values = np.random.default_rng(4).integers(0, 256, (7, 512), dtype=np.uint8)
    encoded = xor_innovations(values)
    assert np.array_equal(xor_inverse(encoded), values)
    assert np.all(xor_inverse(xor_innovations(values)) == values)


def test_timing_excludes_warmup_from_steady_state_observations():
    result = timing_summary(lambda key: jnp.sum(jnp.asarray([1., 2.])), warmups=3, repeats=4)
    assert result["compile_time_ms"] >= 0
    assert result["warmup_calls_excluded"] == 3
    assert len(result["observations_ms"]) == 4
    assert result["steady_state_ms_per_seq"]["mean"] >= 0


def test_whole_sequence_exact_rate_and_byte_accuracy_are_distinct():
    target = np.array([[1, 2, 3], [1, 2, 3]])
    decoded = np.array([[1, 2, 3], [1, 2, 0]])
    metrics = reconstruction_metrics(target, decoded)
    assert metrics["exact_sequence_rate"] == .5
    assert metrics["byte_accuracy"] == 5 / 6


def test_stop_rule_requires_rvq_to_dominate_both_lossless_controls():
    control = {"held_out_bpb": 8., "sample_ngram_jsd": .2, "total_generation_ms_per_seq": 5., "exact_sequence_rate": 1.}
    winner = {"held_out_bpb": 7., "sample_ngram_jsd": .1, "total_generation_ms_per_seq": 4., "exact_sequence_rate": 1.}
    assert pareto_stop({"raw_byte_cached_ar": control, "xor_reversible_ar": control,
                        "rvq_full_ar": winner})["status"] == "proceed_to_tpu_scaling"
    lossy = dict(winner, exact_sequence_rate=.2)
    assert pareto_stop({"raw_byte_cached_ar": control, "xor_reversible_ar": control,
                        "rvq_full_ar": lossy})["status"] == "proceed_to_tpu_scaling"
    losing = dict(control, held_out_bpb=8.1, sample_ngram_jsd=.21)
    assert pareto_stop({"raw_byte_cached_ar": control, "xor_reversible_ar": control,
                        "rvq_full_ar": losing})["status"] == "halt_quantizer_expansion_and_pivot"
