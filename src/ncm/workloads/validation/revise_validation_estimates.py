"""Recalculate historical a1ed801 estimates; never change measured observations.

Raw estimates remain in their immutable study directory and in raw_* columns.
The saved old analytic expression encodes the measured prefill iteration count
linearly, allowing exact reconstruction after removing known decode/verify work.
Only the preregistered historical real128 profile is handled by this revision.
"""
import json
import numpy as np
from ncm.baselines import matched_ar_config
from ncm.flops import (FlopModel,transformer_forward,ar_cached_decode_step,
    ar_cache_prefill_step,ar_cached_generation,ncm_prior_forward,codec_forward)


def shape(config,vocabulary=256):
    return FlopModel(config['width'],config['layers'],config['heads'],config.get('mlp_ratio',4),vocabulary)


def revise(raw,real,revision):
    if not raw['git_commit'].startswith('a1ed801'):
        return dict(applied=False,reason='not historical a1ed801; estimates retained as recorded',components=[])
    if raw['profile']!='real128':raise ValueError('revision applies only to frozen real128 profile')
    metadata={seed:json.loads((real/f'seed-{seed}'/'metadata.json').read_text()) for seed in (200,201,202)}
    ledger=[];contexts={}
    for seed,m in metadata.items():
        if m['teacher_steps']!=8:raise ValueError('historical training budget differs')
        rvq=m['rvq']; lengths=tuple(rvq['lengths']);length=max(lengths)
        teacher=shape(m['teacher']);student=shape(m['transformer'],rvq['codebook_size'])
        # Production RVQ scans a padded [K,L,V] distance tensor, not sum(n_k).
        delta=2*4*(len(lengths)*length-sum(lengths))*rvq['codebook_size']*rvq['dim']
        codec_delta=8*3*delta
        ledger.append(dict(seed=seed,component='codec_training',prompt_len='shared',
            raw_analytic_flops=m['codec_training_analytic_flops'],
            analytic_flops=m['codec_training_analytic_flops']+codec_delta,
            analytic_delta=codec_delta,analytic_label='analytic estimate'))
        for p in (0,16,64):
            q=next(r for r in raw['quality_rows'] if (r['seed'],r['prompt_len'],r['arm'])==(seed,p,'ar_scratch'))
            ac,mismatch=matched_ar_config(q['parameter_match_target'],heads=2,layers=1,max_length=m['capacity'])
            if q['parameter_match_target']+mismatch!=q['parameters']:raise ValueError('AR shape reconstruction mismatch')
            ar=FlopModel(ac.width,ac.layers,ac.heads,ac.mlp_ratio)
            samples=json.loads((real/f'seed-{seed}'/f'prompt-{p}'/'teacher_samples_cost.json').read_text())
            new_generation=ar_cached_generation(teacher,32,m['capacity'],p,length)
            generation_delta=new_generation-samples['generation_analytic_flops']
            ledger.append(dict(seed=seed,prompt_len=p,component='teacher_sample_generation',
                raw_analytic_flops=samples['generation_analytic_flops'],analytic_flops=new_generation,
                analytic_delta=generation_delta,analytic_label='analytic estimate'))
            contexts[seed,p]=dict(meta=m,rvq=rvq,lengths=lengths,length=length,teacher=teacher,
                student=student,ar=ar,encoder_delta=delta,codec_delta=codec_delta,generation_delta=generation_delta)
    for row in raw['training_rows']:
        ctx=contexts[row['seed'],row['prompt_len']]
        row['raw_total_training_flops']=row['total_training_flops']
        row['raw_training_step_analytic_flops']=row['training_step_analytic_flops']
        row['raw_analytic_flops']=row['analytic_flops']
        if row['arm'].startswith('ncm'):
            row['training_step_analytic_flops']+=ctx['encoder_delta']
            row['analytic_flops']=row['training_step_analytic_flops']
            row['total_training_flops']+=ctx['codec_delta']+8*ctx['encoder_delta']
            if row['arm']=='ncm_sequence_distilled':row['total_training_flops']+=ctx['generation_delta']
        row['analytic_model_revision']=revision
    costs={(r['seed'],r['prompt_len'],r['arm']):r for r in raw['training_rows']}
    for row in raw['quality_rows']:
        row['raw_analytic_flops']=row['analytic_flops']
        row['analytic_flops']=costs[row['seed'],row['prompt_len'],row['arm']]['training_step_analytic_flops']
        row['analytic_model_revision']=revision
    for row in raw['checkpoint_rows']:
        ctx=contexts[row['seed'],row['prompt_len']];p=row['prompt_len']
        row['raw_analytic_flops']=row['analytic_flops']
        row['raw_total_training_flops']=row['total_training_flops']
        if row['arm'].startswith('ar_'):
            row['analytic_flops']=ar_cached_generation(ctx['ar'],1,ctx['meta']['capacity'],p,ctx['length'])
        else:
            coarse=0 if row['arm']=='ncm_coarse_0' else 1
            row['analytic_flops']=ncm_prior_forward(ctx['student'],1,ctx['lengths'],ctx['rvq']['dim'],p,coarse,cached=True)
            row['analytic_flops']+=codec_forward(1,ctx['length'],ctx['rvq']['dim'],encoder=False)
            row['total_training_flops']+=ctx['codec_delta']+row['checkpoint']*ctx['encoder_delta']
            if row['arm']=='ncm_sequence_distilled':row['total_training_flops']+=ctx['generation_delta']
        row['analytic_model_revision']=revision
    for row in raw['drafter_rows']:
        ctx=contexts[row['seed'],row['prompt_len']];p=row['prompt_len'];b=row['batch'];d=row['draft_len']
        capacity=ctx['meta']['capacity'];teacher=ctx['teacher']
        original=row['analytic_flops'];row['raw_analytic_flops']=original
        row['raw_analytic_flops_per_round']=row['analytic_flops_per_round']
        if row['drafter']=='cached_teacher_reference':
            row['analytic_flops']=ar_cached_generation(teacher,b,capacity,p,ctx['length'])
            row['raw_analytic_prefill_flops']=row['analytic_prefill_flops']
            row['analytic_prefill_flops']=p*ar_cache_prefill_step(teacher,b,capacity)
        else:
            rounds=row['measured_rounds']
            verify=transformer_forward(teacher,b,p+ctx['length']+d+1)
            if row['drafter']=='ar_distilled':
                decode=ar_cached_decode_step(ctx['ar'],b,capacity)
                old_prefill=ar_cached_decode_step(ctx['ar'],b,capacity,head=False)
                iterations=(original-verify*rounds-d*decode*rounds)/old_prefill
                np.testing.assert_allclose(iterations*30,round(iterations*30),atol=1e-4,rtol=0)
                iterations=round(iterations*30)/30
                if not p*rounds-1e-6<=iterations<=(p+ctx['length'])*rounds+1e-6:
                    raise ValueError('reconstructed prefix iterations outside measured bounds')
                row['measured_prefill_iterations']=iterations
                row['prefill_counter_source']='reconstructed from saved a1ed801 formula; 30 timed observations'
                draft=iterations*ar_cache_prefill_step(ctx['ar'],b,capacity)+d*decode*rounds
                row['analytic_flops']=verify*rounds+draft
                row['analytic_flops_per_round']=row['analytic_flops']/rounds
            else:
                draft=ncm_prior_forward(ctx['student'],b,ctx['lengths'],ctx['rvq']['dim'],p,1,cached=True)
                draft+=codec_forward(b,ctx['length'],ctx['rvq']['dim'],encoder=False)
                # Check the historical expression before replacing its prefill term.
                n=ctx['lengths'][0]
                old_extra=p*(ar_cached_decode_step(ctx['student'],b,p+n,head=False)-ar_cache_prefill_step(ctx['student'],b,p+n))
                np.testing.assert_allclose(original,(verify+draft+old_extra)*rounds,rtol=1e-12)
                row['analytic_flops_per_round']=verify+draft
                row['analytic_flops']=(verify+draft)*rounds
        row['analytic_model_revision']=revision
    return dict(applied=True,raw_measurement_commit=raw['git_commit'],analytic_model_revision=revision,
        observations_changed=False,components=ledger,
        corrections=['cache-only prefill omits final-layer attention/MLP/head',
                     'RVQ distance execution counted on padded K*L grid'],
        convention='MAC=2; fused QKV charged nominally; norm/softmax/backward estimates remain approximate')
