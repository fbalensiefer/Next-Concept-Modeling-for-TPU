"""Isolated repeated timing of identical seed-13 training trajectories."""
import json,time
from pathlib import Path
import numpy as np,jax,jax.numpy as jnp,optax
from ncm.config import experiment_rvq_config
from ncm.tokenizer import MultiscaleTokenizer
from ncm.coarse import optimizer_state,warmup_codebooks,maintain_coarse
from ncm.rvq import init_codebooks
from ncm.coverage import codec_training_distribution
from ncm.scale_experiment import make_codec_step,as_batch,arm_weights,ALL_ARMS
from ncm.training import TrainState
from ncm.maintenance_experiment import minimal_ema,ema_recovery,fixed_binary
root=Path('runs/coarse-codebook-stability');cfg=json.loads((root/'report.json').read_text())['protocol']['config'];rvq=experiment_rvq_config(cfg);d=np.load(root/'teacher_domain.npz');tokens=d['tokens'];train=d['train'];canonical=d['canonical'];candidates,importance=codec_training_distribution(train,16,cfg['codec_sampling']);seed=13;key=jax.random.key(seed);optimizer=optax.adam(cfg['codec_learning_rate']);tok=MultiscaleTokenizer(rvq,256);books=init_codebooks(jax.random.fold_in(key,1),rvq);first=as_batch(tokens[train[np.arange(64)%len(train)]]);params=tok.init(jax.random.fold_in(key,2),first.tokens,first.mask,first.fallback_bytes,first.fallback_mask,books)['params'];optstate=optimizer_state(optimizer.init(params),books,rvq,jax.random.fold_in(key,4));warmups=[as_batch(tokens[np.random.default_rng(np.random.SeedSequence([seed,i,30])).choice(candidates,64)]) for i in range(2)];books,coarse=warmup_codebooks(tok,params,books,optstate['coarse'],warmups);initial=TrainState(params,dict(optstate,coarse=coarse),books,jnp.asarray(0,jnp.int32));batches=[]
for i in range(400):
 ids=np.random.default_rng(np.random.SeedSequence([seed,i,30])).choice(candidates,64);batches.append((as_batch(tokens[ids]),jnp.asarray(canonical[ids]),jnp.asarray(importance[ids])))
treatments={'minimal_ema':minimal_ema,'ema_recovery':ema_recovery,'reference':maintain_coarse,'fixed_binary':fixed_binary};executables={};batch,features,iw=batches[0];w=arm_weights(ALL_ARMS[0],cfg)
for name,fn in treatments.items():
 ex=jax.jit(make_codec_step(cfg,optimizer,maintenance=fn)).lower(initial,batch,features,w,iw).compile();jax.block_until_ready(ex(initial,batch,features,w,iw));executables[name]=ex
jax.block_until_ready((batches,initial));observations=[]
for repeat in range(3):
 for ai,arm in enumerate(ALL_ARMS):
  names=list(treatments);shift=(repeat+ai)%4;names=names[shift:]+names[:shift]
  for name in names:
   state=initial;w=arm_weights(arm,cfg);jax.block_until_ready(w);t=time.perf_counter()
   for batch,features,iw in batches:state,metrics=executables[name](state,batch,features,w,iw)
   jax.block_until_ready((state,metrics));observations.append(dict(repeat=repeat,arm=arm,treatment=name,seconds=time.perf_counter()-t))
summary={name:dict(median_seconds=float(np.median([r['seconds'] for r in observations if r['treatment']==name])),minimum_seconds=min(r['seconds'] for r in observations if r['treatment']==name),maximum_seconds=max(r['seconds'] for r in observations if r['treatment']==name),step_flops=executables[name].cost_analysis().get('flops')) for name in treatments}
result=dict(scope='Seed 13; all four objective arms; three repeats each; 400 original-order updates from identical initialization. No concurrent test or experiment jobs dispatched by this task. Inputs resident, compilation and warmup excluded.',observations=observations,summary=summary)
Path('runs/maintenance-ablation/clean_timing.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
