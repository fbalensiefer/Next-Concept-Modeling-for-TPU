"""Matched maintenance ablation using the archived finite-domain teacher.

Run: python -m ncm.maintenance_experiment
No teacher/prior retraining; evaluation uses exact population code-law oracles.
"""
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from .config import experiment_rvq_config, TransformerConfig
from .coarse import maintain_coarse, optimizer_state, warmup_codebooks
from .coverage import FrozenCodec, codec_training_distribution, encoding_diagnostics, make_decoder_repair_step
from .dependence import codewords, conditional_tc, hard_joint
from .generation import PriorState
from .grids import align, grid_matrices
from .rvq import ema_update, init_codebooks
from .scale_experiment import ALL_ARMS, arm_weights, as_batch, make_codec_step, save_state, state_bytes
from .tokenizer import MultiscaleTokenizer
from .training import TrainState


def minimal_ema(books, state, counts, sums, latent, mask, rvq, *, phase=0):
    """Ordinary EMA, same diagnostic clock, without replacement work."""
    del latent, mask
    proposal = ema_update(books, counts, sums, rvq, phase=phase)
    eligible = ((jnp.asarray(phase) == 0) | (rvq.levels <= 2)) & (counts[0].sum() > 0)
    usage = rvq.coarse.usage_decay * state.hard_counts + (1-rvq.coarse.usage_decay) * counts[0]
    probability = counts[0] / jnp.maximum(counts[0].sum(), 1)
    pp = jnp.exp(-jnp.sum(probability * jnp.log(jnp.maximum(probability, 1e-30))))
    pp = jnp.where(counts[0].sum() > 0, pp, 0)
    watch = eligible & (state.step >= max(rvq.coarse.temp_anneal_steps,
                                         rvq.coarse.beta_warmup_steps,
                                         rvq.coarse.dead_code_grace_steps))
    updated = state.replace(step=state.step+1,
        hard_counts=jnp.where(eligible, usage, state.hard_counts),
        monitored_steps=state.monitored_steps+watch.astype(jnp.int32),
        collapsed_steps=state.collapsed_steps+(watch & (jnp.sum(counts[0]>0)<rvq.codebook_size)).astype(jnp.int32),
        min_hard_perplexity=jnp.where(watch,jnp.minimum(state.min_hard_perplexity,pp),state.min_hard_perplexity))
    return proposal, updated, {'coarse_active_codes':(counts[0]>0).sum(),
        'coarse_hard_perplexity':pp,'coarse_dead_replacements':jnp.asarray(0),
        'coarse_usage_min':updated.hard_counts.min()}


def ema_recovery(books, state, counts, sums, latent, mask, rvq, *, phase=0):
    return maintain_coarse(books,state,counts,sums,latent,mask,
        replace(rvq,coarse=replace(rvq.coarse,coarse_codebook_lr_scale=1.)),phase=phase)


def exact_metrics(tokenizer, codec, tokens, p):
    batch=as_batch(tokens)
    o=tokenizer.apply({'params':codec.params},batch.tokens,batch.mask,batch.fallback_bytes,batch.fallback_mask,codec.codebooks)
    cfg=tokenizer.config
    codes=np.concatenate([np.asarray(o.rvq.indices[k,:,:n]) for k,n in enumerate(cfg.lengths)],axis=1)
    joint=np.asarray(hard_joint(jnp.asarray(codes),jnp.asarray(p),cfg.lengths,cfg.codebook_size),np.float64)
    joint/=joint.sum()
    words=codewords(cfg.lengths,cfg.codebook_size)
    q=np.ones(len(words),np.float64); offset=0
    for n in cfg.lengths:
        for i,word in enumerate(words):
            context=np.all(words[:,:offset]==word[:offset],axis=1); mass=joint[context].sum()
            for j in range(offset,offset+n):
                q[i]*=joint[context & (words[:,j]==word[j])].sum()/mass if mass else 1/cfg.codebook_size
        offset+=n
    assert np.isclose(q.sum(),1.)
    _,up=grid_matrices(cfg)
    representation=jnp.zeros((len(words),cfg.max_length,cfg.dim));offset=0
    for k,n in enumerate(cfg.lengths):
        vectors=codec.codebooks.embeddings[k][words[:,offset:offset+n]]
        v=jnp.pad(vectors,((0,0),(0,cfg.max_length-n),(0,0)))
        valid=jnp.broadcast_to(jnp.arange(cfg.max_length)<n,(len(words),cfg.max_length))
        representation+=align(v,valid,up[k])[0];offset+=n
    decoded=np.asarray(tokenizer.apply({'params':codec.params},representation,jnp.ones(representation.shape[:2],bool),method=tokenizer.decode).argmax(-1))
    valid=np.all((decoded>=65)&(decoded<=66),axis=1)
    dest=np.where(valid,(decoded-65)@2**np.arange(cfg.max_length-1,-1,-1),len(tokens))
    def tv(law):return float(abs(np.bincount(dest,weights=law,minlength=len(tokens)+1)-np.append(p,0)).sum()/2)
    pred=np.asarray(o.logits.argmax(-1));coarse=np.asarray(o.rvq.indices[0,:,:cfg.lengths[0]])
    mass=np.bincount(coarse.ravel(),weights=np.repeat(p,cfg.lengths[0])/cfg.lengths[0],minlength=cfg.codebook_size)
    positive=joint>0
    kl=float(np.sum(joint[positive]*np.log(joint[positive]/q[positive])))
    # Float64 verification avoids cancellation of many float32 marginal entropies.
    table=joint.reshape((cfg.codebook_size,)*sum(cfg.lengths))
    def marginal_entropy(kept):
        marginal=table.sum(axis=tuple(i for i in range(sum(cfg.lengths)) if i not in kept))
        return -float(np.sum(marginal*np.log(np.maximum(marginal,1e-300))))
    start=0; entropy_tc=0.
    for count in cfg.lengths:
        prefix=tuple(range(start))
        entropy_tc+=sum(marginal_entropy(prefix+(start+j,)) for j in range(count))
        entropy_tc-=marginal_entropy(tuple(range(start+count)))+(count-1)*marginal_entropy(prefix)
        start+=count
    assert np.isclose(kl,entropy_tc,atol=1e-12,rtol=1e-12)
    return dict(encoding_diagnostics(codes,p),byte_reconstruction=float(p@(pred==tokens).mean(-1)),
        sequence_reconstruction=float(p@np.all(pred==tokens,axis=1)),active_coarse_codes=int((mass>0).sum()),
        coarse_perplexity=float(np.exp(-np.sum(mass*np.log(np.maximum(mass,1e-30))))),
        factorized_oracle_kl_nats=kl,factorized_oracle_tv=tv(q),joint_oracle_tv=tv(joint),
        factorized_missing_teacher_mass=float(p[np.bincount(dest,weights=q,minlength=len(tokens)+1)[:len(tokens)]==0].sum()),
        factorized_outside_mass=float(np.bincount(dest,weights=q,minlength=len(tokens)+1)[-1]))


def fixed_binary(books, state, counts, sums, latent, mask, rvq, *, phase=0):
    """Fixed two-vector coarse quantizer; equivalent to an affine binary threshold.

    The encoder adapts, coarse endpoints stay at matched initialization, and
    fine codebooks retain EMA. This is not a full multilevel FSQ implementation.
    """
    proposal, tracking, metrics = minimal_ema(books,state,counts,sums,latent,mask,rvq,phase=phase)
    proposal=proposal.replace(embeddings=proposal.embeddings.at[0].set(books.embeddings[0]),
        counts=proposal.counts.at[0].set(books.counts[0]),sums=proposal.sums.at[0].set(books.sums[0]))
    return proposal, tracking, metrics


def run(binary=False):
    root=Path('reports/coarse-codebook-stability');out=Path('runs/binary-maintenance-comparison' if binary else 'runs/maintenance-ablation');out.mkdir(parents=True,exist_ok=True)
    archived=json.loads((root/'report.json').read_text());cfg=archived['protocol']['config'];rvq=experiment_rvq_config(cfg)
    domain=np.load(Path('reports/maintenance-ablation/teacher_domain.npz'));tokens=domain['tokens'];p=domain['probability'];canonical=domain['canonical'];train=domain['train']
    candidates,importance=codec_training_distribution(train,len(tokens),cfg['codec_sampling'])
    treatments=({'ema_recovery':ema_recovery,'fixed_binary':fixed_binary} if binary else {'minimal_ema':minimal_ema,'ema_recovery':ema_recovery,'reference':maintain_coarse})
    protocol={'config':cfg,'seeds':list(range(13,18)),'treatments':list(treatments),
        'fixed_objectives':'All entropy, temperature and commitment settings retained. Stage one changes centroid rate and recovery; stage two freezes coarse endpoints at the common initialization.',
        'selection_context': 'Stage two follows observed stage-one collapse under minimal EMA; recovery is the simplest stable candidate, not proven superior to reference.' if binary else 'Stage one',
        'evaluation':'Before/after 200 identical decoder replay updates; exact population oracle, no prior training.',
        'selection':'Require all 20 sequence reconstruction >=.98, no collisions, full output support via factorized oracle, coarse perplexity >=1.5, collapsed fraction <=.05; then compare paired oracle errors. No automatic winner when fidelity gates fail.',
        'timing':'Precomputed on-device batches; synchronized first discarded update for warmup, then synchronized whole training window. Compilation, preparation, initialization and replay reported separately. Treatment order rotated by seed and arm.',
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).parent.glob('*.py'))},
        'teacher_domain_sha256':hashlib.sha256(Path('reports/maintenance-ablation/teacher_domain.npz').read_bytes()).hexdigest(),
        'versions':{'jax':jax.__version__,'numpy':np.__version__,'python':platform.python_version()},'backend':jax.default_backend()}
    path=out/'protocol.json'
    if path.exists():raise ValueError('Use a fresh output directory; experiment already exists')
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for f in Path(__file__).parent.glob('*.py'):(snapshot/f.name).write_bytes(f.read_bytes())
    optimizer=optax.adam(cfg['codec_learning_rate']);tokenizer=MultiscaleTokenizer(rvq,256)
    functions={name:jax.jit(make_codec_step(cfg,optimizer,maintenance=fn)) for name,fn in treatments.items()}
    repair_opt=optax.adam(cfg['decoder_repair_learning_rate']);repair_fn=jax.jit(make_decoder_repair_step(rvq.dim,repair_opt))
    rows=[]
    for seed in range(13,18):
        key=jax.random.key(seed);books=init_codebooks(jax.random.fold_in(key,1),rvq)
        first=as_batch(tokens[train[np.arange(cfg['batch_size'])%len(train)]])
        params=tokenizer.init(jax.random.fold_in(key,2),first.tokens,first.mask,first.fallback_bytes,first.fallback_mask,books)['params']
        optstate=optimizer_state(optimizer.init(params),books,rvq,jax.random.fold_in(key,4))
        t=time.perf_counter();warmups=[]
        for i in range(rvq.coarse.warmup_batches):
            ids=np.random.default_rng(np.random.SeedSequence([seed,i,30])).choice(candidates,cfg['batch_size']);warmups.append(as_batch(tokens[ids]))
        books,coarse=warmup_codebooks(tokenizer,params,books,optstate['coarse'],warmups)
        initial=TrainState(params,dict(optstate,coarse=coarse),books,jnp.asarray(0,jnp.int32));jax.block_until_ready(initial);init_seconds=time.perf_counter()-t
        t=time.perf_counter();batches=[];replay_ids=[]
        for i in range(cfg['codec_steps']):
            ids=np.random.default_rng(np.random.SeedSequence([seed,i,30])).choice(candidates,cfg['batch_size']);batches.append((as_batch(tokens[ids]),jnp.asarray(canonical[ids]),jnp.asarray(importance[ids])))
        distinct=np.unique(train)
        for i in range(cfg['decoder_repair_steps']):replay_ids.append(jnp.asarray(np.random.default_rng(np.random.SeedSequence([seed,i,32])).integers(0,len(distinct),cfg['batch_size'])))
        jax.block_until_ready((batches,replay_ids));prep_seconds=time.perf_counter()-t
        for ai,arm in enumerate(ALL_ARMS):
            order=list(treatments);rotation=(seed+ai)%len(treatments);order=order[rotation:]+order[:rotation]
            for treatment in order:
                directory=out/f'seed-{seed}'/arm/treatment;directory.mkdir(parents=True)
                weights=arm_weights(arm,cfg);batch,features,iw=batches[0]
                t=time.perf_counter();exe=functions[treatment].lower(initial,batch,features,weights,iw).compile();compile_seconds=time.perf_counter()-t
                jax.block_until_ready(exe(initial,batch,features,weights,iw))
                state=initial;t=time.perf_counter()
                for batch,features,iw in batches:state,metrics=exe(state,batch,features,weights,iw)
                jax.block_until_ready((state,metrics));train_seconds=time.perf_counter()-t
                save_state(directory/'codec.msgpack',state)
                frozen=FrozenCodec(state.params,state.codebooks);before=exact_metrics(tokenizer,frozen,tokens,p)
                replay_batch=as_batch(tokens[distinct]);enc=tokenizer.apply({'params':state.params},replay_batch.tokens,replay_batch.mask,replay_batch.fallback_bytes,replay_batch.fallback_mask,state.codebooks)
                repair=PriorState(state.params['decoder'],repair_opt.init(state.params['decoder']),jnp.asarray(0,jnp.int32));ids=replay_ids[0]
                t=time.perf_counter();rex=repair_fn.lower(repair,enc.representation[ids],replay_batch.tokens[ids]).compile();replay_compile_seconds=time.perf_counter()-t
                jax.block_until_ready(rex(repair,enc.representation[ids],replay_batch.tokens[ids]));t=time.perf_counter()
                for ids in replay_ids:repair,loss=rex(repair,enc.representation[ids],replay_batch.tokens[ids])
                jax.block_until_ready((repair,loss));replay_seconds=time.perf_counter()-t
                frozen=FrozenCodec(dict(state.params,decoder=repair.params),state.codebooks);after=exact_metrics(tokenizer,frozen,tokens,p);save_state(directory/'codec_repaired.msgpack',frozen)
                tracking=state.opt_state['coarse'];row=dict(seed=seed,arm=arm,treatment=treatment,before=before,after=after,
                    initial_state_sha256=hashlib.sha256(state_bytes(initial)).hexdigest(),
                    codec_seconds=train_seconds,compile_seconds=compile_seconds,initialization_seconds_shared=init_seconds,
                    data_preparation_seconds_shared=prep_seconds,replay_seconds=replay_seconds,replay_compile_seconds=replay_compile_seconds,
                    codec_step_flops=exe.cost_analysis().get('flops'),replay_step_flops=rex.cost_analysis().get('flops'),
                    replacements=int(tracking.replacements.sum()),collapsed_fraction=float(tracking.collapsed_steps)/max(int(tracking.monitored_steps),1))
                (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
                (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps({'seed':seed,'arm':arm,'treatment':treatment,'seconds':train_seconds,'sequence':after['sequence_reconstruction'],'oracle_tv':after['factorized_oracle_tv'],'pp':after['coarse_perplexity']}),flush=True)
    summary={}
    for name in treatments:
        rr=[r for r in rows if r['treatment']==name];failures=[{'seed':r['seed'],'arm':r['arm']} for r in rr if r['after']['sequence_reconstruction']<.98 or r['after']['population_unique_encodings']<16 or r['after']['factorized_missing_teacher_mass']>0 or r['after']['coarse_perplexity']<1.5 or r['collapsed_fraction']>.05]
        summary[name]={'models':len(rr),'failures':failures,'mean':{k:float(np.mean([r['after'][k] for r in rr])) for k in ['sequence_reconstruction','factorized_oracle_tv','factorized_oracle_kl_nats','coarse_perplexity']},'median_codec_seconds':float(np.median([r['codec_seconds'] for r in rr])),'codec_step_flops':sorted(set(r['codec_step_flops'] for r in rr))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binary',action='store_true')
    run(binary=parser.parse_args().binary)
