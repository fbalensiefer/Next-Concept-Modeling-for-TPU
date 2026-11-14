"""Exact source-space controls; these are mathematical oracles, not trained models."""
import json
from pathlib import Path

import numpy as np

from ncm.domain_validation import DOMAINS, source_law


def scheduled_law(tokens, p, levels):
    """Product of true per-symbol conditionals under a prescribed parallel schedule."""
    q=np.ones(len(p));past=[]
    for level in levels:
        for i,word in enumerate(tokens):
            context=np.all(tokens[:,past]==word[past],axis=1)
            for position in level:
                q[i]*=p[context & (tokens[:,position]==word[position])].sum()/p[context].sum()
        past+=level
    assert np.isclose(q.sum(),1.)
    return q


def midpoint_levels(length):
    levels=[];intervals=[(0,length)]
    while intervals:
        level=[];next_intervals=[]
        for left,right in intervals:
            mid=(left+right-1)//2;level.append(mid)
            if left<mid:next_intervals.append((left,mid))
            if mid+1<right:next_intervals.append((mid+1,right))
        levels.append(level);intervals=next_intervals
    return levels




def innovations(name, bits):
    """Known-structure binary bijections; not learned from samples."""
    bits=np.asarray(bits,np.int32)
    if name=='persistent_markov':
        return np.column_stack([bits[:,0],bits[:,1:] ^ bits[:,:-1]])
    if name=='noisy_parity':
        return np.column_stack([bits[:,:-1],bits.sum(axis=1)%2])
    raise ValueError(name)


def invert_innovations(name, values):
    values=np.asarray(values,np.int32)
    if name=='persistent_markov':return np.bitwise_xor.accumulate(values,axis=1)
    if name=='noisy_parity':
        return np.column_stack([values[:,:-1],values[:,-1] ^ (values[:,:-1].sum(axis=1)%2)])
    raise ValueError(name)


if __name__=='__main__':
    result={}
    for name,length,_ in DOMAINS:
        tokens,p=source_law(name,length)
        schedules={'independent':[list(range(length))]}
        if name=='persistent_markov':schedules['midpoint_bridge']=midpoint_levels(length)
        if name=='noisy_parity':schedules['last_bit_conditional']=[list(range(length-1)),[length-1]]
        row=dict(population=len(p),entropy_bits=float(-np.sum(p*np.log2(p))),schedules={})
        for schedule,levels in schedules.items():
            q=scheduled_law(tokens,p,levels)
            row['schedules'][schedule]=dict(levels=levels,serial_rounds=len(levels),
                tv=float(abs(p-q).sum()/2),kl_nats=float(np.sum(p*np.log(p/q))))
        if name in ('persistent_markov','noisy_parity'):
            values=innovations(name,tokens-65)
            assert np.array_equal(invert_innovations(name,values),tokens-65)
            assert len(np.unique(values,axis=0))==len(tokens)
            marginal=p@values
            product=np.prod(np.where(values,marginal,1-marginal),axis=1)
            row['invertible_innovations']=dict(marginal_one_probabilities=marginal.tolist(),tv=float(abs(p-product).sum()/2),max_probability_error=float(abs(p-product).max()),bijective=True)
        result[name]=row
    target=Path('reports/invertible-baselines/results.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
