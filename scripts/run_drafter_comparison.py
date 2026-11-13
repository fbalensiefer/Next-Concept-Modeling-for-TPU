#!/usr/bin/env python3
"""Frozen local drafter controls and fair continuation accounting; no downloads."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from run_decisive_benchmark import measure,compiler_flops,batch
from ncm.accounting import continuation_mask,matched_latency_rows,speculative_batch_observation
from ncm.baselines import CausalAR,matched_ar_config,parameter_count,sample_conditioned,shifted_inputs
from ncm.config import NCMConfig,RVQConfig,TransformerConfig
from ncm.data import load_text
from ncm.flops import (FlopModel,transformer_forward,ar_training_step,ar_cached_generation,
    ar_cached_decode_step,ncm_codec_step,ncm_prior_forward,ncm_prior_step,codec_forward,speculative_round)
from ncm.generation import encode,level_geometry
from ncm.generator import SharedLevelGenerator
from ncm.grids import align,grid_matrices
from ncm.losses import cross_entropy
from ncm.speculative import DistributionAdapter,speculative_generate,ar_adapter
from ncm.tokenizer import MultiscaleTokenizer
from ncm.training import NCM,initialize,make_train_step
from ncm.validation_io import identity,write_json,write_csv,paired_bootstrap

LABELS=dict(compiler_label='compiler estimate',analytic_label='analytic estimate')
PINNED={'train':'9b9b640d78534c383583b720752ed3cde1599e49f8d6c5f1ebe17c18b7481093',
        'validation':'c6c72478d8e39d2b71fdfe923d43dd08da32d4dc59378f47ce221d63a8b811e6'}


def corpus(length,max_prompt):
    arrays=[]; provenance=[]
    for split in ('train','validation'):
        path=Path('runs/real-text-pilot')/(split+'.jsonl')
        data=load_text(path,length+max_prompt,expected_sha256=PINNED[split])
        ids=np.flatnonzero(data.mask[:,:max_prompt+1].all(1))
        chosen=np.r_[ids[:31],ids[-1]]
        if len(set(chosen))!=32:raise ValueError('32 distinct eligible windows required')
        arrays.append((jnp.asarray(data.tokens[chosen]),jnp.asarray(data.mask[chosen])))
        provenance.append(dict(split=split,path=str(path),sha256=data.sha256,windows=chosen.tolist(),
            continuation_bytes=int(data.mask[chosen,max_prompt:].sum()),
            continuation_sha256=hashlib.sha256(data.tokens[chosen,max_prompt:].tobytes()+data.mask[chosen,max_prompt:].tobytes()).hexdigest()))
    return arrays,provenance


def save_checkpoint(path,params):
    path.write_bytes(serialization.to_bytes(dict(git_commit=identity()['git_commit'],params=params)))


def fit(params,objective,optimizer,data,key,steps,checkpoints):
    x,mask=data; opt=optimizer.init(params)
    def step(params,opt,key):
        ids=jax.random.randint(key,(4,),0,len(x))
        value,grads=jax.value_and_grad(objective)(params,x[ids],mask[ids])
        updates,opt=optimizer.update(grads,opt,params)
        return optax.apply_updates(params,updates),opt,value
    lowered=jax.jit(step).lower(params,opt,key)
    estimate=compiler_flops(lowered); compiled=lowered.compile(); saved={}
    for index in range(1,steps+1):
        params,opt,_=compiled(params,opt,jax.random.fold_in(key,1000+index))
        if index in checkpoints:saved[index]=params
    jax.block_until_ready(params)
    return params,saved,estimate


def ar_bpb(model,params,x,mask,p):
    @jax.jit
    def score(x,mask):
        logits=model.apply({'params':params},shifted_inputs(x),mask)
        selected=continuation_mask(mask,p)
        nll=optax.softmax_cross_entropy_with_integer_labels(logits,x)
        return jnp.where(selected,nll,0).sum(),selected.sum()
    total=count=0.
    for start in range(0,len(x),4):
        n,b=score(x[start:start+4],mask[start:start+4]);total+=float(n);count+=int(b)
    return total/count/np.log(2),int(count)


def run_seed(seed,profile,out,progress):
    smoke=profile=='smoke'; length=8 if smoke else 128; max_prompt=2 if smoke else 64
    prompts=(2,) if smoke else (0,16,64)
    drafts=(2,4,8) if smoke else (2,4,8,16)
    steps=1 if smoke else 8; checkpoints=(1,) if smoke else (2,4,8)
    (train,held),provenance=corpus(length,max_prompt)
    tx,tm=train; hx,hm=held
    key=jax.random.key(seed); optimizer=optax.adam(.0003)
    config=TransformerConfig(8,2,1); rvq=RVQConfig((max(1,length//16),max(1,length//4),length),4,4)
    codec_model=NCM(rvq,config,256); tokenizer=MultiscaleTokenizer(rvq,256)
    initial_batch=batch(tx[:1,max_prompt:],tm[:1,max_prompt:])
    state=initialize(codec_model,optimizer,key,initial_batch)
    raw=make_train_step(codec_model,optimizer); codec_step=jax.jit(raw)
    cbatch=batch(tx[:4,max_prompt:],tm[:4,max_prompt:])
    codec_compiler=compiler_flops(codec_step.lower(state,cbatch,jnp.asarray(0),key))
    codec_analytic=ncm_codec_step(4,length,4,parameter_count(state.params),levels=3,codes=4,lengths=rvq.lengths)
    codec_analytic+=3*ncm_prior_forward(FlopModel(8,1,2,vocabulary=4),4,(length,)*3,4,0)
    for i in range(steps):
        ids=jax.random.randint(jax.random.fold_in(key,1000+i+1),(4,),0,len(tx))
        state,_=codec_step(state,batch(tx[ids,max_prompt:],tm[ids,max_prompt:]),jnp.asarray(0),jax.random.fold_in(key,i))
    cp=state.params['MultiscaleTokenizer_0']; books=state.codebooks
    codec_cost=steps*codec_analytic; codec_compiler_cost=steps*codec_compiler
    capacity=max_prompt+length+max(drafts)+1
    teacher=CausalAR(TransformerConfig(16,2,2),capacity)
    tp=teacher.init(key,shifted_inputs(tx[:1]),tm[:1])['params']; teacher_shape=FlopModel(16,2,2)
    def teacher_objective(params,x,mask):
        return cross_entropy(teacher.apply({'params':params},shifted_inputs(x),mask),x,continuation_mask(mask,max_prompt))
    tp,_,teacher_compiler=fit(tp,teacher_objective,optimizer,train,key,steps,checkpoints)
    teacher_analytic=ar_training_step(teacher_shape,4,max_prompt+length,parameter_count(tp))
    teacher_cost=steps*teacher_analytic; teacher_compiler_cost=steps*teacher_compiler
    directory=out/f'seed-{seed}';directory.mkdir(parents=True,exist_ok=True)
    save_checkpoint(directory/'codec.msgpack',dict(params=cp,books=books))
    save_checkpoint(directory/'teacher.msgpack',tp)
    write_json(directory/'metadata.json',dict(seed=seed,profile=profile,corpus=provenance,
        rvq=asdict(rvq),transformer=asdict(config),teacher=asdict(teacher.config),capacity=capacity,
        teacher_parameters=parameter_count(tp),teacher_steps=steps,
        codec_training_compiler_flops=codec_compiler_cost,codec_training_analytic_flops=codec_cost,
        teacher_training_compiler_flops=teacher_compiler_cost,teacher_training_analytic_flops=teacher_cost,**LABELS))
    _,up=grid_matrices(rvq)
    def decode(representation,mask,decoder):
        params=dict(cp,decoder=decoder)
        return tokenizer.apply({'params':params},representation,mask,method=tokenizer.decode)
    def soft_decode(logits,masks,decoder):
        representation=jnp.zeros((masks.shape[1],length,4))
        for k in range(3):
            vectors=jax.nn.softmax(logits[k],-1)@books.embeddings[k]
            value,_=align(vectors,masks[k],up[k]);representation+=value
        return decode(representation,masks[-1],decoder)
    output_rows=[]; quality_rows=[]; checkpoint_rows=[]; training_rows=[]; matched=[]
    for p in prompts:
        pd=directory/f'prompt-{p}';pd.mkdir(exist_ok=True)
        x=tx[:,max_prompt-p:];mask=tm[:,max_prompt-p:]
        vx=hx[:,max_prompt-p:];vm=hm[:,max_prompt-p:]
        teacher_bpb,denominator=ar_bpb(teacher,tp,vx,vm,p)
        gens={k:SharedLevelGenerator(rvq,config,NCMConfig(k,p)) for k in (0,1)}
        target=encode(codec_model,cp,books,initial_batch)
        initial_gp={k:g.init(key,books,target.mask,target.indices,prompt_ids=x[:1,:p])['params'] for k,g in gens.items()}
        target_params=parameter_count(cp)+parameter_count(books.embeddings)+parameter_count(initial_gp[1])
        ac,_=matched_ar_config(target_params,heads=2,layers=1,max_length=capacity)
        small=CausalAR(ac,capacity); small_shape=FlopModel(ac.width,ac.layers,ac.heads)
        ap=small.init(key,shifted_inputs(x[:1]),mask[:1])['params']
        ar_parameters={}; ar_saved={}; train_costs={}; comp_costs={}; analytic_steps={}; compiler_steps={}
        curves={}
        for name,distilled in (('ar_scratch',False),('ar_distilled',True)):
            def objective(params,bx,bm):
                logits=small.apply({'params':params},shifted_inputs(bx),bm)
                selected=continuation_mask(bm,p);loss=cross_entropy(logits,bx,selected)
                if distilled:
                    target_logits=teacher.apply({'params':tp},shifted_inputs(bx),bm)
                    probabilities=jax.nn.softmax(target_logits,-1)
                    kl=jnp.sum(probabilities*(jax.nn.log_softmax(target_logits)-jax.nn.log_softmax(logits)),-1)
                    loss=.5*loss+.5*jnp.where(selected,kl,0).sum()/selected.sum()
                return loss
            params,saved,compiler=fit(ap,objective,optimizer,(x,mask),key,steps,checkpoints)
            ar_parameters[name]=params;ar_saved[name]=saved
            analytic=ar_training_step(small_shape,4,p+length,parameter_count(params))
            if distilled:analytic+=transformer_forward(teacher_shape,4,p+length)
            analytic_steps[name]=analytic;compiler_steps[name]=compiler
            train_costs[name]=steps*analytic+(teacher_cost if distilled else 0)
            comp_costs[name]=steps*compiler+(teacher_compiler_cost if distilled else 0)
            save_checkpoint(pd/(name+'.msgpack'),params)
        teacher_warning=abs(teacher_bpb-ar_bpb(small,ar_parameters['ar_scratch'],vx,vm,p)[0])<=.3
        priors={}; prior_saved={}; decoders={}
        for coarse in (0,1):
            gen=gens[coarse];name=f'ncm_coarse_{coarse}'
            def objective(gp,bx,bm):
                targets=encode(codec_model,cp,books,batch(bx[:,p:],bm[:,p:]))
                result=gen.apply({'params':gp},books,targets.mask,targets.indices,prompt_ids=bx[:,:p])
                logits=soft_decode(result.logits,targets.mask,cp['decoder'])
                teacher_logits=teacher.apply({'params':tp},shifted_inputs(bx),bm)[:,p:]
                probabilities=jax.nn.softmax(teacher_logits,-1)
                kl=jnp.sum(probabilities*(jax.nn.log_softmax(teacher_logits)-jax.nn.log_softmax(logits)),-1)
                return cross_entropy(result.logits,targets.indices,targets.mask)+jnp.where(bm[:,p:],kl,0).sum()/bm[:,p:].sum()
            gp,saved,compiler=fit(initial_gp[coarse],objective,optimizer,(x,mask),key,steps,checkpoints)
            priors[name]=gp;prior_saved[name]=saved;decoders[name]=cp['decoder']
            storage=rvq.lengths if p or coarse else (length,)*3
            analytic=ncm_prior_step(FlopModel(8,1,2,vocabulary=4),4,storage,4,p,parameter_count(gp),
                ar_coarse_levels=coarse,codes=4,teacher=teacher_shape,teacher_length=p+length)
            analytic_steps[name]=analytic;compiler_steps[name]=compiler
            train_costs[name]=codec_cost+teacher_cost+steps*analytic
            comp_costs[name]=codec_compiler_cost+teacher_compiler_cost+steps*compiler
            save_checkpoint(pd/(name+'.msgpack'),gp)
        # Paired teacher-sampled ablation; neither encoder nor codebooks is updated.
        teacher_samples=jax.jit(lambda rng:sample_conditioned(teacher,tp,x[:32,:p],length,rng))(jax.random.fold_in(key,300+p))
        sampled_x=jnp.concatenate([x[:32,:p],teacher_samples],1);sampled_mask=jnp.ones_like(sampled_x,bool)
        generation_analytic=ar_cached_generation(teacher_shape,32,capacity,p,length)
        generation_compiler=compiler_flops(jax.jit(lambda rng:sample_conditioned(teacher,tp,x[:32,:p],length,rng)).lower(key))
        # Compiler loop body estimate is reported separately from executed-loop analytic estimate.
        gen=gens[1]
        def sequence_objective(payload,bx,bm):
            targets=encode(codec_model,cp,books,batch(bx[:,p:],bm[:,p:]))
            result=gen.apply({'params':payload['prior']},books,targets.mask,targets.indices,prompt_ids=bx[:,:p])
            decoded=decode(result.representation,bm[:,p:],payload['decoder'])
            return cross_entropy(result.logits,targets.indices,targets.mask)+cross_entropy(decoded,bx[:,p:],bm[:,p:])
        payload,saved,seq_compiler=fit(dict(prior=initial_gp[1],decoder=cp['decoder']),sequence_objective,
            optimizer,(sampled_x,sampled_mask),key,steps,checkpoints)
        sequence_saved=saved
        name='ncm_sequence_distilled';priors[name]=payload['prior'];decoders[name]=payload['decoder']
        prior_saved[name]={i:params['prior'] for i,params in saved.items()}
        seq_analytic=ncm_prior_step(FlopModel(8,1,2,vocabulary=4),4,rvq.lengths,4,p,
            parameter_count(payload),ar_coarse_levels=1,codes=4,decoder_trainable=True)
        analytic_steps[name]=seq_analytic;compiler_steps[name]=seq_compiler
        train_costs[name]=codec_cost+teacher_cost+generation_analytic+steps*seq_analytic
        comp_costs[name]=codec_compiler_cost+teacher_compiler_cost+generation_compiler+steps*seq_compiler
        save_checkpoint(pd/(name+'.msgpack'),payload)
        save_checkpoint(pd/'teacher_samples.msgpack',dict(tokens=teacher_samples,prompt=x[:32,:p]))
        write_json(pd/'teacher_samples_cost.json',dict(seed=seed,prompt_len=p,samples=32,
            generation_compiler_flops=generation_compiler,generation_analytic_flops=generation_analytic,**LABELS))

        def ncm_bpb(gen,gp,decoder):
            @jax.jit
            def score(bx,bm):
                targets=encode(codec_model,cp,books,batch(bx[:,p:],bm[:,p:]))
                result=gen.apply({'params':gp},books,targets.mask,targets.indices,prompt_ids=bx[:,:p])
                logits=decode(result.representation,bm[:,p:],decoder)
                code_ce=optax.softmax_cross_entropy_with_integer_labels(result.logits,jnp.maximum(targets.indices,0))
                token_ce=optax.softmax_cross_entropy_with_integer_labels(logits,bx[:,p:])
                return jnp.where(targets.mask,code_ce,0).sum()+jnp.where(bm[:,p:],token_ce,0).sum(),bm[:,p:].sum()
            total=count=0.
            for start in range(0,len(vx),4):
                n,b=score(vx[start:start+4],vm[start:start+4]);total+=float(n);count+=int(b)
            assert count==denominator
            return total/count/np.log(2)

        def ncm_logits(gen,payload,prompt,rng):
            valid=jnp.ones((len(prompt),length),bool);geometry=level_geometry(rvq,valid)
            result=gen.apply({'params':payload['prior']},books,geometry,jnp.zeros_like(geometry,jnp.int32),
                prompt_ids=prompt,teacher_forcing=False,temperature=1.,use_cache=True,rngs={'sample':rng})
            return decode(result.representation,valid,payload['decoder'])
        all_params=dict(ar_parameters,**priors)
        for arm in ('ar_scratch','ar_distilled','ncm_coarse_0','ncm_coarse_1','ncm_sequence_distilled'):
            is_ar=arm.startswith('ar_');coarse=0 if arm=='ncm_coarse_0' else 1
            arm_gen=gens[coarse]
            bpb=ar_bpb(small,all_params[arm],vx,vm,p)[0] if is_ar else ncm_bpb(arm_gen,priors[arm],decoders[arm])
            kind='exact' if is_ar else 'upper_bound'
            qrow=dict(seed=seed,prompt_len=p,arm=arm,heldout_bpb=bpb,bpb_kind=kind,
                continuation_bytes=denominator,continuation_sha256=provenance[1]['continuation_sha256'],
                available_prompt_bytes=p,parameters=parameter_count(all_params[arm]) if is_ar else parameter_count(cp)+parameter_count(priors[arm])+parameter_count(books.embeddings),
                parameter_match_target=target_params,teacher_bpb=teacher_bpb,teacher_steps=steps,teacher_parameters=parameter_count(tp),
                teacher_warning='teacher within 0.3 BPB of small AR' if teacher_warning else '',**LABELS)
            quality_rows.append(qrow)
            training_rows.append(dict(**qrow,total_training_flops=train_costs[arm],
                total_training_compiler_estimate=comp_costs[arm],
                training_step_compiler_flops=compiler_steps[arm],training_step_analytic_flops=analytic_steps[arm],
                teacher_training_included=arm!='ar_scratch',codec_training_included=not is_ar,
                teacher_forward_in_step=arm in ('ar_distilled','ncm_coarse_0','ncm_coarse_1')))
            curves[arm]=[]
            saved=ar_saved[arm] if is_ar else prior_saved[arm]
            for checkpoint,params in saved.items():
                decoder=payload['decoder'] if arm=='ncm_sequence_distilled' else cp['decoder']
                if arm=='ncm_sequence_distilled':
                    # Keep decoder from the same optimization checkpoint.
                    decoder=sequence_saved[checkpoint]['decoder']
                checkpoint_bpb=ar_bpb(small,params,vx,vm,p)[0] if is_ar else ncm_bpb(arm_gen,params,decoder)
                if is_ar:
                    sample=lambda prompt,rng,params=params:sample_conditioned(small,params,prompt,length,rng)
                    infer_analytic=ar_cached_generation(small_shape,1,capacity,p,length)
                else:
                    def sample(prompt,rng,params=params,decoder=decoder,arm_gen=arm_gen):
                        latent,token=jax.random.split(rng)
                        return jax.random.categorical(token,ncm_logits(arm_gen,dict(prior=params,decoder=decoder),prompt,latent))
                    ns=rvq.lengths if p or coarse else (length,)*3
                    infer_analytic=ncm_prior_forward(FlopModel(8,1,2,vocabulary=4),1,ns,4,p,coarse,cached=True)+codec_forward(1,length,4,encoder=False)
                timing=measure(sample,(vx[:1,:p],key),5,30)
                row=dict(seed=seed,prompt_len=p,arm=arm,checkpoint=checkpoint,bpb=checkpoint_bpb,bpb_kind=kind,
                    median_ms=timing['median_ms'],compiler_flops=timing['compiler_flops'],analytic_flops=infer_analytic,
                    total_training_flops=train_costs[arm]-(steps-checkpoint)*analytic_steps[arm],**LABELS)
                checkpoint_rows.append(row);curves[arm].append(row)
            print(f'quality seed={seed} P={p} arm={arm} BPB={bpb:.4f} {kind}',flush=True)
        matched.extend(dict(seed=seed,prompt_len=p,**r,**LABELS) for r in matched_latency_rows(curves))
        teacher_reference={}
        for size in (1,16,128):
            prompt=vx[jnp.arange(size)%len(vx),:p]
            timing=measure(lambda prompt,rng:sample_conditioned(teacher,tp,prompt,length,rng),(prompt,key),5,30)
            teacher_reference[size]=timing['median_ms']
            output_rows.append(dict(seed=seed,drafter='cached_teacher_reference',draft_len=0,prompt_len=p,batch=size,
                acceptance=1.,tokens_per_pass=1.,per_position_acceptance=[],median_ms=timing['median_ms'],
                speedup_vs_cached_teacher=1.,compiler_flops=timing['compiler_flops'],
                analytic_flops=ar_cached_generation(teacher_shape,size,capacity,p,length),
                bpb_kind='exact',teacher_bpb=teacher_bpb,teacher_warning=teacher_warning,**LABELS))
        teacher_adapter=ar_adapter(teacher,tp)
        for drafter in ('ncm','ar_distilled','ncm_sequence_distilled'):
            for draft_len in drafts:
                if drafter=='ar_distilled':
                    proposal=ar_adapter(small,ar_parameters['ar_distilled'],draft=True)
                    dp=proposal.params
                else:
                    arm='ncm_coarse_1' if drafter=='ncm' else drafter
                    dp=dict(prior=priors[arm],decoder=decoders[arm])
                    def draft_apply(params,buffer,position,d,rng):
                        u=jax.lax.dynamic_slice(buffer,(position-p,),(p,))[None]
                        return ncm_logits(gens[1],params,u,rng)[0,:d]
                    proposal=DistributionAdapter(dp,draft_apply)
                def sampled(dp,tp,prompt,rng):
                    adapter=DistributionAdapter(dp,proposal.apply_fn,proposal.draft_fn)
                    verifier=DistributionAdapter(tp,teacher_adapter.apply_fn)
                    keys=jax.random.split(rng,len(prompt))
                    return jax.vmap(lambda row,k:speculative_generate(adapter,verifier,row,draft_len,length,k))(prompt,keys)
                for size in (1,16,128):
                    prompt=vx[jnp.arange(size)%len(vx),:p]
                    observe=lambda stats:speculative_batch_observation(stats,p,length)
                    timing=measure(sampled,(dp,tp,prompt,key),5,30,collect=observe)
                    observations=timing.pop('observations')
                    passes=float(np.mean([v['mean_rounds'] for v in observations]))
                    batch_rounds=float(np.mean([v['batch_rounds'] for v in observations]))
                    accepted=np.sum([v['accepted'] for v in observations],axis=0)
                    proposed=np.sum([v['proposed'] for v in observations],axis=0)
                    mean_prefix=sum(v['prefix_sum'] for v in observations)/sum(v['sequence_rounds'] for v in observations)
                    verify=transformer_forward(teacher_shape,size,p+length+draft_len+1)
                    if drafter=='ar_distilled':
                        prefill_steps=float(np.mean([v['prefill_iterations'] for v in observations]))
                        total_draft=prefill_steps*ar_cached_decode_step(small_shape,size,capacity,head=False)
                        total_draft+=batch_rounds*draft_len*ar_cached_decode_step(small_shape,size,capacity)
                        draft_cost=total_draft/batch_rounds
                    else:
                        draft_cost=ncm_prior_forward(FlopModel(8,1,2,vocabulary=4),size,rvq.lengths,4,p,1,cached=True)+codec_forward(size,length,4,encoder=False)
                    row=dict(seed=seed,drafter=drafter,draft_len=draft_len,prompt_len=p,batch=size,
                        acceptance=float(accepted.sum()/proposed.sum()),tokens_per_pass=length/passes,
                        per_position_acceptance=(accepted/np.maximum(proposed,1)).tolist(),
                        acceptance_kind='accepted-prefix survival fraction per eligible draft position',
                        median_ms=timing['median_ms'],speedup_vs_cached_teacher=teacher_reference[size]/timing['median_ms'],
                        compiler_flops=timing['compiler_flops'],**speculative_round(draft_cost,verify,batch_rounds),
                        bpb_kind='exact_teacher',teacher_bpb=teacher_bpb,teacher_warning=teacher_warning,
                        measured_rounds_kind='mean of maximum batched rounds across all 30 timed observations',
                        mean_sequence_rounds=passes,batch_tokens_per_teacher_call=size*length/batch_rounds,
                        timing_observations=30,
                        mean_prefix_length=mean_prefix,latency_note='coarse KV cache measured' if drafter!='ar_distilled' else 'prefix cache rebuilt each round',
                        compiler_label='compiler estimate')
                    output_rows.append(row)
                    progress['drafter_rows'].append(row)
                    write_csv(out/'drafter_comparison.csv',progress['drafter_rows'])
                    print(f'timing seed={seed} P={p} D={draft_len} B={size} {drafter} {row["median_ms"]:.2f} ms',flush=True)
        progress['quality_rows'].extend(quality_rows);quality_rows=[]
        progress['training_rows'].extend(training_rows);training_rows=[]
        progress['checkpoint_rows'].extend(checkpoint_rows);checkpoint_rows=[]
        progress['matched_rows'].extend(matched);matched=[]
        # Reference rows were not appended in the inner speculative loop.
        progress['drafter_rows'].extend(r for r in output_rows if r['drafter']=='cached_teacher_reference')
        output_rows=[]
        flush(out,progress)
        jax.clear_caches()
    return dict(seed=seed,status='complete')


def flush(out,progress):
    commit=identity()['git_commit']
    for key in ('drafter_rows','quality_rows','training_rows','checkpoint_rows','matched_rows'):
        for row in progress[key]:row.setdefault('git_commit',commit)
    for key,filename in [('drafter_rows','drafter_comparison.csv'),('training_rows','bpb_vs_train_flops.csv'),
                         ('checkpoint_rows','checkpoint_curves.csv'),('matched_rows','latency_at_matched_bpb.csv'),('quality_rows','quality.csv')]:
        write_csv(out/filename,progress[key])
    write_json(out/'results.json',progress)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','real128'],default='smoke')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    progress=dict(profile=args.profile,seeds=[200] if args.profile=='smoke' else [200,201,202],
        drafter_rows=[],quality_rows=[],training_rows=[],checkpoint_rows=[],matched_rows=[],outcomes=[],
        limitations=['All NCM BPB values are upper bounds, not exact quality measures.',
          'Compiler estimates are static costs; analytic estimates multiply explicit measured rounds.',
          'Teacher verification recomputes the full prefix; cached teacher is the fair reference.',
          'NCM fixed-bucket drafter generates the whole 128-token latent block even for shorter drafts.',
          'Only 32 local real-text windows per split and eight training steps: feasibility, not convergence.',
          'No TPU evidence. Three-seed confidence intervals are exploratory.'])
    for seed in progress['seeds']:
        try:progress['outcomes'].append(run_seed(seed,args.profile,out,progress))
        except Exception as error:
            progress['outcomes'].append(dict(seed=seed,status='failed',error=repr(error)))
            flush(out,progress);raise
        flush(out,progress)
    contrasts=[]
    for p in sorted({r['prompt_len'] for r in progress['quality_rows']}):
        for a,b in [('ar_distilled','ar_scratch'),('ncm_coarse_1','ar_distilled'),('ncm_coarse_0','ncm_coarse_1')]:
            records={arm:{r['seed']:r['heldout_bpb'] for r in progress['quality_rows'] if r['prompt_len']==p and r['arm']==arm} for arm in (a,b)}
            seeds=progress['seeds']
            if all(s in records[a] and s in records[b] for s in seeds):
                contrasts.append(dict(prompt_len=p,contrast=a+' minus '+b,seed_list=seeds,
                    **paired_bootstrap([records[a][s] for s in seeds],[records[b][s] for s in seeds]),
                    comparison_kind='exact' if a.startswith('ar_') else 'upper_bound_comparison',**LABELS))
    write_json(out/'paired_statistics.json',dict(contrasts=contrasts))
    flush(out,progress)


if __name__=='__main__':main()
