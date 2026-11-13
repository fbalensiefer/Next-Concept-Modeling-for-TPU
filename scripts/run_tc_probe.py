#!/usr/bin/env python3
"""Preregistered learned TC probe gate. No text TC claim when validation fails."""
import argparse
from pathlib import Path
import numpy as np
from ncm.probe import source_law,exact_conditional_tc,train_probe
from ncm.validation_io import identity,write_csv,write_json


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in (200,201,202):
        for length in (8,10):
            lengths=(length//2,length-length//2)
            for source_index,source in enumerate(('markov','parity','copied_bit','hmm','constant_mixture','uniform')):
                words,law=source_law(source,length)
                exact=exact_conditional_tc(words,law,lengths)
                for size in (256,512,2048):
                    # Independent deterministic streams, never resample to hide overlaps.
                    stream=source_index*100000+length*1000+size
                    rng=np.random.default_rng(np.random.SeedSequence([seed,stream,0]))
                    train_ids=rng.choice(len(words),size=size,p=law)
                    rng=np.random.default_rng(np.random.SeedSequence([seed,stream,1]))
                    eval_ids=rng.choice(len(words),size=4096,p=law)
                    overlap=len(set(train_ids)&set(eval_ids))
                    for k in range(2):
                        parallel,_=train_probe(words[train_ids],words[eval_ids],lengths,k,seed,causal=False)
                        ar,_=train_probe(words[train_ids],words[eval_ids],lengths,k,seed,causal=True)
                        estimate=parallel-ar; error=abs(estimate-exact[k]); tolerance=max(.02,.1*exact[k])
                        row=dict(seed=seed,source=source,length=length,training_size=size,level=k,
                            exact_tc_nats=exact[k],delta_nll_nats=estimate,parallel_nll_nats=parallel,
                            ar_nll_nats=ar,absolute_error=error,tolerance=tolerance,
                            numerical_gate=error<=tolerance,identity_overlap=overlap,
                            disjoint_identity_gate=overlap==0)
                        rows.append(row)
                    print(f'probe seed={seed} L={length} source={source} N={size}',flush=True)
                    write_json(output/'probe_results.json',dict(rows=rows,status='running'))
    numerical=float(np.mean([r['numerical_gate'] for r in rows]))
    disjoint=all(r['disjoint_identity_gate'] for r in rows)
    passed=numerical>=.9 and disjoint
    write_json(output/'probe_results.json',dict(rows=rows,numerical_pass_fraction=numerical,
        disjoint_identity_gate=disjoint,real_text_gate=passed,status='complete',seeds=[200,201,202]))
    write_csv(output/'probe_results.csv',rows)
    coarse=sum(max(r['delta_nll_nats'],0) for r in rows if r['level']==0)
    fine=sum(max(r['delta_nll_nats'],0) for r in rows if r['level']==1)
    fraction=coarse/max(coarse+fine,1e-30)
    text=f'''# TC probe decision

git_commit: {identity()['git_commit']}

Numerical gate pass fraction: {numerical:.6f}; required 0.90.
Strict disjoint-sequence-identity gate: {disjoint}.
Real-text gate: {passed}.

Synthetic pooled estimated TC share: coarse {fraction:.6f}, fine {1-fraction:.6f}.
Synthetic >=30% coarse-share decision: {fraction>=.3}.
These pooled finite-model estimates are biased diagnostics, not real-text TC.
'''
    if not passed:
        text+='\nNo TC claim is permitted on real text. Real-text code probing is stopped.\nCopied-bit has only two distinct sequences; identity-disjoint train/eval\nwith the same nondegenerate source law is impossible. Overlaps and numerical\nfailures remain in the table. No additional hybrid models are trained.\n'
    else:
        text+='\nEnumerable gate passes. No text codes were supplied; real-text TC remains untested.\n'
    (output/'tc_probe_decision.md').write_text(text)
    return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/validation-controls/tc-probe')
    args=parser.parse_args();run(args.output)
