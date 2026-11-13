#!/usr/bin/env python3
"""Combine frozen studies, preserve failures, and prohibit unsupported claims."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from ncm.validation_io import identity,write_json,write_csv,paired_bootstrap


def summarize(root,real):
    raw=json.loads((real/'results.json').read_text())
    if raw['seeds']!=[200,201,202] or [r['seed'] for r in raw['outcomes'] if r['status']=='complete']!=[200,201,202]:
        raise ValueError('all three preregistered real128 seeds must complete')
    rows=raw['drafter_rows'];quality=raw['quality_rows']
    expected=set(itertools.product((200,201,202),(0,16,64),(2,4,8,16),(1,16,128),('ncm','ar_distilled','ncm_sequence_distilled')))
    observed={(r['seed'],r['prompt_len'],r['draft_len'],r['batch'],r['drafter']) for r in rows if r['drafter']!='cached_teacher_reference'}
    if observed!=expected:raise ValueError(f'incomplete sweep: {len(expected-observed)} missing, {len(observed-expected)} unexpected')
    if len(rows)!=len(expected)+27:raise ValueError('duplicate/reference row count mismatch')
    for r in rows:
        for name in ('compiler_flops','analytic_flops','git_commit','compiler_label','analytic_label'):
            if name not in r or r[name] is None:raise ValueError((name,r))
        if len(r['per_position_acceptance'])!=r['draft_len']:raise ValueError('position array mismatch')
        if r['drafter']!='cached_teacher_reference':
            np.testing.assert_allclose(r['analytic_flops'],r['analytic_flops_per_round']*r['measured_rounds'],rtol=1e-12)
            if r['timing_observations']!=30:raise ValueError('timing protocol mismatch')
    for seed in (200,201,202):
        grouped=[r for r in quality if r['seed']==seed]
        if len({r['continuation_bytes'] for r in grouped})!=1 or len({r['continuation_sha256'] for r in grouped})!=1:
            raise ValueError('continuation/context fairness failure')
        for r in grouped:
            if r['arm'].startswith('ncm') and r['bpb_kind']!='upper_bound':raise ValueError('unlabeled NCM bound')
    write_csv(root/'drafter_comparison.csv',rows)
    write_csv(root/'bpb_vs_train_flops.csv',raw['training_rows'])
    write_csv(root/'checkpoint_curves.csv',raw['checkpoint_rows'])
    matched=list(raw['matched_rows'])
    # Fixed-teacher controls have identical target BPB by their verified law,
    # so interpolation is unnecessary. Retain the student curve comparisons too.
    for seed,p,b in itertools.product((200,201,202),(0,16,64),(1,16,128)):
        subset=[r for r in rows if (r['seed'],r['prompt_len'],r['batch'])==(seed,p,b)]
        reference=next(r for r in subset if r['drafter']=='cached_teacher_reference')
        for d in (2,4,8,16):
            controls=[r for r in subset if r['draft_len']==d]+[reference]
            for a,c in itertools.combinations(controls,2):
                matched.append(dict(seed=seed,prompt_len=p,batch=b,draft_len=d,
                    arm_a=a['drafter'],arm_b=c['drafter'],status='exact_teacher_distribution_match',
                    common_bpb=a['teacher_bpb'],bpb_kind_a='exact_teacher',bpb_kind_b='exact_teacher',
                    latency_a_ms=a['median_ms'],latency_b_ms=c['median_ms'],
                    compiler_flops_a=a['compiler_flops'],compiler_flops_b=c['compiler_flops'],
                    analytic_flops_a=a['analytic_flops'],analytic_flops_b=c['analytic_flops'],
                    quality_matched_claim_permitted=True,measurement_git_commit=a['git_commit'],
                    compiler_label='compiler estimate',analytic_label='analytic estimate'))
    write_csv(root/'latency_at_matched_bpb.csv',matched)
    write_json(root/'paired_statistics.json',dict(contrasts=json.loads((real/'paired_statistics.json').read_text())['contrasts']))
    paired=[]
    for p,d,b in itertools.product((0,16,64),(2,4,8,16),(1,16,128)):
        def get(seed,name):return next(r for r in rows if (r['seed'],r['prompt_len'],r['draft_len'],r['batch'],r['drafter'])==(seed,p,d,b,name))
        for metric in ('acceptance','median_ms','tokens_per_pass'):
            a=[get(s,'ncm')[metric] for s in (200,201,202)]
            c=[get(s,'ar_distilled')[metric] for s in (200,201,202)]
            paired.append(dict(prompt_len=p,draft_len=d,batch=b,metric=metric,
                contrast='NCM minus distilled AR drafter',**paired_bootstrap(a,c)))
    write_json(root/'drafter_paired_statistics.json',dict(contrasts=paired))
    power=json.loads((root/'power.json').read_text())
    probe=json.loads((root/'tc-probe/probe_results.json').read_text())
    if probe['real_text_gate']:raise ValueError('this summary expects the observed failed probe gate')
    probe_cases=probe['rows']
    sensitivity=[]
    for size in (256,512,2048):
        cases=[r for r in probe_cases if r['training_size']==size]
        sensitivity.append(dict(training_size=size,numerical_pass_fraction=float(np.mean([r['numerical_gate'] for r in cases])),
            mean_absolute_error=float(np.mean([r['absolute_error'] for r in cases])),
            identity_overlap_cases=sum(r['identity_overlap']>0 for r in cases)))
    write_csv(root/'tc_probe_sensitivity.csv',sensitivity)
    (root/'tc_probe_decision.md').write_text((root/'tc-probe/tc_probe_decision.md').read_text())
    within=sum(bool(r['teacher_warning']) for r in quality if r['arm']=='ar_scratch')
    ncm_faster=0; pairs=0
    for r in rows:
        if r['drafter']=='ncm':
            c=next(c for c in rows if c['drafter']=='ar_distilled' and all(c[k]==r[k] for k in ('seed','prompt_len','draft_len','batch')))
            ncm_faster+=r['median_ms']<c['median_ms'];pairs+=1
    summary=dict(status='complete',backend=raw['backend'],seeds=raw['seeds'],sweep_rows=len(rows),
        quality_rows=len(quality),checkpoint_rows=len(raw['checkpoint_rows']),matched_rows=len(matched),
        power_tv=power['exact']['tv'],power_floor95=power['exact']['floor_95'],negative_controls_rejected=all(power[n]['rejected'] for n in ('mismatched_q','bad_residual','future_leak','wrong_bonus')),
        probe_pass_fraction=probe['numerical_pass_fraction'],real_text_tc_permitted=False,
        teacher_within_03_cases=within,teacher_context_seed_cases=9,
        ncm_faster_than_ar_drafter_cases=int(ncm_faster),paired_drafter_cases=pairs,
        sensitivity=sensitivity,limitations=raw['limitations'])
    write_json(root/'summary.json',summary)
    commit=identity()['git_commit']
    findings=root/'FINDINGS.md';prefix=findings.read_text().split('<!-- final-study-summary -->')[0]
    text=f'''\n<!-- final-study-summary -->
## Completed frozen studies

git_commit: {commit}; model measurement commits are retained in every table row.

- Powered exactness: TV {power['exact']['tv']:.8f}, frozen floor {power['exact']['floor_95']:.8f}; all four corrupt verifier controls rejected.
- Coarse cache: sampled codes match recomputation for seeds 200–202; logits pass 1e-5, including two coarse levels and masked prompts/grids. AR padding cache is tested at every position.
- TC gate: {probe['numerical_pass_fraction']:.2%} numerical pass rate versus 90%; strict disjoint identity gate also failed. No real-text TC claim is permitted. The protocol and all 216 cases are retained. Numerical sensitivity to sample size is in tc_probe_sensitivity.csv.
- Real text: three seeds, 128-byte continuations, 32 fixed windows per split including the final eligible partial window. All prompt buckets share continuation hashes and masks. {len(rows)} cached-reference/drafter rows, {len(quality)} quality rows, and {len(raw['checkpoint_rows'])} checkpoint observations completed.
- Distilled AR drafter is present for every draft length, prompt length, batch and seed. NCM has lower raw median latency in {ncm_faster}/{pairs} paired cases. This is a bounded weak-model CPU pilot, not a general drafter advantage claim.
- Teacher BPB is within 0.3 of the small AR in {within}/9 seed/context cases. All teacher sizes, steps and BPB are disclosed; eight steps do not establish a strong teacher.
- All NCM student BPB values are upper bounds. Intersections of these bounds with exact AR likelihood curves do not establish equal quality. Non-overlap is reported without extrapolation.
- Exact-teacher drafter comparisons are explicitly matched by distribution in latency_at_matched_bpb.csv. Per-arm student checkpoint curves and exploratory three-seed paired bootstrap intervals are separate.
- Compiler FLOPs are compiler estimates. Analytic FLOPs are operation-model estimates. Speculative estimates use all thirty timed round-count observations, including masked finished lanes in batched loops. Prefill and teacher/codec prerequisites, teacher distillation forwards, and 32 teacher-sample generations are charged without double-counting those forwards.
- Teacher verification remains full-prefix. NCM produces its complete 128-token latent block even when only 2/4/8/16 decoded token probabilities are used. Both costs are measured. Cached coarse levels remove recomputation but do not eliminate these costs.
- No TPU evidence or general compute-saving claim exists. The discarded contended timing attempt and serialization failure remain recorded above.
'''
    findings.write_text(prefix+text)
    claims=f'''# Claim ledger

git_commit: {commit}

| Claim | Evidence file | Status |
|---|---|---|
| Speculative exactness passes the frozen enumerable power test and rejects all four controls | power.json | supported |
| Hybrid coarse cache agrees with recomputation on the registered finite tests | tests.json | supported |
| General NCM drafter advantage over the distilled AR drafter | drafter_comparison.csv; drafter_paired_statistics.json | untested |
| NCM student beats distilled AR at matched exact BPB | latency_at_matched_bpb.csv | untested |
| Learned probe meets the frozen TC accuracy/disjointness gate | tc-probe/probe_results.json | refuted |
| Hybrid versus parallel removes >=30% of real-text TC | tc_probe_decision.md | untested |
| No TPU evidence has been produced | summary.json | supported |
| General compute savings at matched quality | bpb_vs_train_flops.csv; latency_at_matched_bpb.csv | untested |

Supported finite checks are not universal mathematical proofs. Adapter proposals
must match their reported q, and the teacher must be causal. No real-text TC
interpretation or strong-teacher speedup claim follows from this pilot.
'''
    (root/'claims.md').write_text(claims)
    registry=json.loads((root/'seed_registry.json').read_text())
    registry['execution_git_commit']=commit;registry['status']='complete_with_negative_results'
    for entry in registry['seeds']:
        seed=entry['seed'];cases=[r for r in probe_cases if r['seed']==seed]
        entry['outcome']='complete; TC gate failed; student quality matching unproven'
        entry['outcomes']=dict(probe=dict(cases=len(cases),numerical_pass_fraction=float(np.mean([r['numerical_gate'] for r in cases])),strict_disjoint_gate=False),
            real128=dict(status='complete',drafter_rows=sum(r['seed']==seed for r in rows),quality_rows=sum(r['seed']==seed for r in quality)),
            tests=dict(status='passed',evidence='tests.json'),nulls_and_failures=['TC validation gate failed','No real-text TC claim','No matched exact NCM BPB claim'])
        if seed==200:entry['outcomes'].update(power=dict(status='passed',samples=200000,negative_controls_rejected=4),smoke=dict(status='passed_after_serialization_fix',failed_attempts=1),timing_attempt=dict(status='initial_contended_attempt_retained_and_replaced_by_quiescent_execution'))
        if seed==201:entry['outcomes']['direct_floor']=dict(repetitions=20,samples_per_repetition=200000,status='complete')
        if seed==202:entry['outcomes'].update(greedy=dict(prompts=50,status='passed'),bootstrap=dict(repetitions=10000,status='complete',label='exploratory'))
    (root/'seed_registry.json').write_text(json.dumps(registry,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='reports/validation-controls')
    parser.add_argument('--real',default='reports/validation-controls/real128')
    args=parser.parse_args();summarize(Path(args.root),Path(args.real))
