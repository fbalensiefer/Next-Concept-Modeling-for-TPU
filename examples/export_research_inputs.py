"""Export compact inference tables/feature diagnostics from local checkpoints.

Run once when original ignored runs are available. Public replay scripts use
these compressed arrays without requiring any trained checkpoint.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from flax import serialization
from ncm.config import experiment_rvq_config
from ncm.tokenizer import MultiscaleTokenizer
from ncm.rvq import CodebookState
from ncm.dependency_validation import codec_tables
from ncm.scale_experiment import as_batch

root=Path('runs/fresh-domain-validation');cfg=json.loads((root/'protocol.json').read_text())['config']
rows=[r for r in json.loads((root/'results.json').read_text()) if r['treatment']=='reference'];arrays={};metadata=[]
for index,row in enumerate(rows):
    domain=row['domain'];length=row['length'];d=np.load(root/domain/'domain.npz')
    for key in ('tokens','probability','train','validation'):arrays[domain+'_'+key]=d[key]
    path=root/domain/f"seed-{row['seed']}"/row['arm']/'reference/codec_repaired.msgpack'
    tokenizer=MultiscaleTokenizer(experiment_rvq_config(dict(cfg,lengths=[1,length])),256)
    codes,words,destinations,reconstructed=codec_tables(tokenizer,serialization.msgpack_restore(path.read_bytes()),d['tokens'])
    prefix=f'm{index:03d}';arrays[prefix+'_codes']=codes.astype(np.int8);arrays[prefix+'_destinations']=destinations.astype(np.int16);arrays[prefix+'_reconstructed']=reconstructed.astype(np.int16)
    metadata.append(dict(prefix=prefix,domain=domain,seed=row['seed'],arm=row['arm'],population_codec_failures=row['failures'],codec_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
folder=Path('reports/dependency-validation');np.savez_compressed(folder/'codec_tables.npz',**arrays)
(folder/'codec_tables.json').write_text(json.dumps(dict(schema=1,models=metadata,sha256=hashlib.sha256((folder/'codec_tables.npz').read_bytes()).hexdigest()),indent=2)+'\n')
root=Path('runs/coarse-codebook-stability');cfg=json.loads((root/'report.json').read_text())['protocol']['config'];d=np.load(root/'teacher_domain.npz');teacher=serialization.msgpack_restore((root/'teacher.msgpack').read_bytes())['params'];arrays={key:d[key] for key in ('tokens','probability','canonical','mean','projection','scale')};arrays['head_kernel']=teacher['head']['kernel'][:,65:67];arrays['head_bias']=teacher['head']['bias'][65:67]
# Recover unprojected teacher features with the actual causal network.
from ncm.autoregressive import AutoregressiveBytes,shifted_inputs
from ncm.config import TransformerConfig
batch=as_batch(d['tokens']);model=AutoregressiveBytes(TransformerConfig(cfg['teacher_width'],2,1),4)
arrays['teacher_hidden']=np.asarray(model.apply({'params':teacher},shifted_inputs(batch.tokens),batch.mask,method=model.represent))
latents=[];quantized=[];metadata=[];tokenizer=MultiscaleTokenizer(experiment_rvq_config(cfg),256)
for row in json.loads((root/'report.json').read_text())['results']:
    if row['seed'] not in range(13,18):continue
    path=root/f"seed-{row['seed']}"/row['arm']/'codec_repaired.msgpack';state=serialization.msgpack_restore(path.read_bytes());books=CodebookState(**{k:jnp.asarray(v) for k,v in state['codebooks'].items()})
    o=tokenizer.apply({'params':state['params']},batch.tokens,batch.mask,batch.fallback_bytes,batch.fallback_mask,books)
    latents.append(np.asarray(o.latent));quantized.append(np.asarray(o.rvq.quantized));metadata.append(dict(seed=row['seed'],arm=row['arm'],codec_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
arrays['latents']=np.stack(latents);arrays['quantized']=np.stack(quantized);folder=Path('reports/alignment-diagnosis');np.savez_compressed(folder/'diagnostic_inputs.npz',**arrays)
(folder/'inputs.json').write_text(json.dumps(dict(models=metadata,sha256=hashlib.sha256((folder/'diagnostic_inputs.npz').read_bytes()).hexdigest(),teacher_sha256=hashlib.sha256((root/'teacher.msgpack').read_bytes()).hexdigest()),indent=2)+'\n')
print('Exported 120 codec tables and 20 alignment diagnostics.')
