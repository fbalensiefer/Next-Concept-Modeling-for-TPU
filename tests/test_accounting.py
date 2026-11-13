import jax.numpy as jnp
import numpy as np
from ncm.accounting import bpb_from_sums,continuation_mask,matched_latency_rows
from ncm.flops import FlopModel,ar_training_step,transformer_forward,speculative_round


def test_bpb_denominators_same_context_and_masked_tail():
    tokens=jnp.array([[9,8,1,2,3,4],[7,6,4,3,0,0]])
    mask=jnp.array([[1,1,1,1,1,1],[1,1,1,1,0,0]],bool)
    p=2
    # Both arms consume precisely tokens[:,:p] as prompt and score this selector.
    selected=continuation_mask(mask,p)
    ar,b_ar=bpb_from_sums(12.,mask,p)
    ncm,b_ncm=bpb_from_sums(15.,mask,p)
    assert int(b_ar)==int(b_ncm)==6
    np.testing.assert_array_equal(np.asarray(tokens)[np.asarray(selected)],[1,2,3,4,4,3])
    np.testing.assert_allclose(float(ar)*6*np.log(2),12,rtol=1e-6)
    np.testing.assert_allclose(float(ncm)*6*np.log(2),15,rtol=1e-6)


def test_quality_matching_refuses_extrapolation_and_bound_claim():
    curves={'a':[dict(bpb=2,median_ms=1,bpb_kind='exact'),dict(bpb=3,median_ms=2,bpb_kind='exact')],
            'b':[dict(bpb=4,median_ms=1,bpb_kind='exact')],
            'c':[dict(bpb=2.5,median_ms=1,bpb_kind='upper_bound')]}
    rows=matched_latency_rows(curves)
    assert next(r for r in rows if r['arm_a']=='a' and r['arm_b']=='b')['status']=='no_overlap'
    assert next(r for r in rows if r['arm_a']=='a' and r['arm_b']=='c')['status']=='upper_bound_not_quality_match'


def test_flop_estimates_include_prerequisites_and_rounds():
    c=FlopModel(8,1,2)
    forward=transformer_forward(c,4,128)
    training=ar_training_step(c,4,128,100)
    assert training>3*forward
    round=speculative_round(100,200,3.5)
    assert round['analytic_flops']==1050
    assert round['analytic_label']=='analytic estimate'


def test_report_serializes_numpy_statistics(tmp_path):
    import json
    from ncm.validation_io import write_json
    path=tmp_path/'report.json'
    write_json(path,dict(warning=np.bool_(True),count=np.int64(3),values=np.array([.1,.2])))
    report=json.loads(path.read_text())
    assert report['warning'] is True and report['count']==3
    assert report['values']==[.1,.2] and report['git_commit']
