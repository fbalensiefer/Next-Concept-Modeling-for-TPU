"""Shared continuation accounting and non-extrapolating quality comparisons."""
import itertools
import numpy as np
import jax.numpy as jnp


def continuation_mask(mask,prompt_len):
    return mask & (jnp.arange(mask.shape[-1])[None]>=prompt_len)


def bpb_from_sums(nll_sum,mask,prompt_len):
    count=continuation_mask(mask,prompt_len).sum()
    return nll_sum/jnp.maximum(count,1)/jnp.log(2.),count


def matched_latency_rows(curves):
    """Interpolate observed curves, with explicit likelihood-kind restrictions."""
    rows=[]
    for a,b in itertools.combinations(sorted(curves),2):
        ca,cb=curves[a],curves[b]
        low=max(min(x['bpb'] for x in ca),min(x['bpb'] for x in cb))
        high=min(max(x['bpb'] for x in ca),max(x['bpb'] for x in cb))
        comparable=all(x['bpb_kind']=='exact' for x in ca+cb)
        if low>high:
            rows.append(dict(arm_a=a,arm_b=b,status='no_overlap',bpb_kind_a=ca[0]['bpb_kind'],bpb_kind_b=cb[0]['bpb_kind']))
            continue
        common=(low+high)/2
        def interp(curve):
            values=sorted((x['bpb'],x['median_ms']) for x in curve)
            return float(np.interp(common,[x[0] for x in values],[x[1] for x in values]))
        rows.append(dict(arm_a=a,arm_b=b,status='overlap' if comparable else 'upper_bound_not_quality_match',
            common_bpb=common,latency_a_ms=interp(ca),latency_b_ms=interp(cb),
            bpb_kind_a=ca[0]['bpb_kind'],bpb_kind_b=cb[0]['bpb_kind'],
            quality_matched_claim_permitted=comparable))
    return rows
