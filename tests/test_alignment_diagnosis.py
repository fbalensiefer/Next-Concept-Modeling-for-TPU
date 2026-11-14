from ncm.workloads.research.alignment_diagnosis import diagnose


def test_causal_target_is_blind_to_last_bit_despite_remaining_information():
    result=diagnose();inv=result['invariants']
    assert inv['first_position_target_max_range']==0
    assert inv['last_bit_pair_target_max_difference']==0
    assert .42<inv['last_bit_conditional_entropy_bits']<.44
    assert inv['pca_teacher_predictive_kl_nats']<1e-5
    assert len(result['models'])==20
