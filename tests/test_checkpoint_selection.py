from ncm.checkpoint_selection import score_candidate


def test_checkpoint_score_flags_codebook_collapse_and_ber_regression():
    rows = {
        32: {'byte_error_rate': 0.20, 'exact_match_rate': 0.01,
             'codebook_perplexity': [4.0, 8.0]},
        128: {'byte_error_rate': 0.18, 'exact_match_rate': 0.0,
              'codebook_perplexity': [1.2, 8.0]},
        512: {'byte_error_rate': 0.25, 'exact_match_rate': 0.0,
              'codebook_perplexity': [5.0, 8.0]},
    }
    result = score_candidate({'length_bytes': 512, 'codebook_size': 16}, rows)
    assert result['collapsed_stages'] == []
    assert result['ber_scaling_stable'] is False
    assert result['valid_loss'] is None


def test_checkpoint_score_flags_perplexity_near_one():
    rows = {32: {'byte_error_rate': 0.1, 'exact_match_rate': 0.1,
                 'codebook_perplexity': [1.01, 5.0]}}
    result = score_candidate({'length_bytes': 32, 'codebook_size': 16}, rows)
    assert result['collapsed_stages'] == [0]
    assert result['codebook_healthy'] is False
