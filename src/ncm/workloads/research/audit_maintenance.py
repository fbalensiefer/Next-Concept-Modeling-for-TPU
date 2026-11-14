import json,hashlib
from pathlib import Path
import numpy as np,jax.numpy as jnp
from flax import serialization
from ncm.config import experiment_rvq_config
from ncm.tokenizer import MultiscaleTokenizer
from ncm.rvq import CodebookState
from ncm.coverage import FrozenCodec
from ncm.maintenance_experiment import exact_metrics
archive=Path('runs/coarse-codebook-stability');domain=np.load(archive/'teacher_domain.npz');cfg=json.loads((archive/'report.json').read_text())['protocol']['config'];tok=MultiscaleTokenizer(experiment_rvq_config(cfg),256)
for study in ['maintenance-ablation','binary-maintenance-comparison']:
 root=Path('runs')/study;rows=json.loads((root/'results.json').read_text());audits=[]
 for row in rows:
  directory=root/f"seed-{row['seed']}"/row['arm']/row['treatment'];s=serialization.msgpack_restore((directory/'codec_repaired.msgpack').read_bytes());codec=FrozenCodec(s['params'],CodebookState(**{k:jnp.asarray(v) for k,v in s['codebooks'].items()}));metrics=exact_metrics(tok,codec,domain['tokens'],domain['probability']);assert abs(metrics['factorized_oracle_tv']-row['after']['factorized_oracle_tv'])<1e-9
  audits.append(dict(seed=row['seed'],arm=row['arm'],treatment=row['treatment'],metrics=metrics))
 (root/'supplementary_support_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
 print(study,'missing support',sum(x['metrics']['factorized_missing_teacher_mass']>0 for x in audits),'outside',sum(x['metrics']['factorized_outside_mass']>0 for x in audits),flush=True)
a=json.loads(Path('runs/maintenance-ablation/results.json').read_text());b=json.loads(Path('runs/binary-maintenance-comparison/results.json').read_text());aa={(r['seed'],r['arm']):r for r in a if r['treatment']=='ema_recovery'}
print('max repeated recovery metric discrepancy',max(abs(r['after']['factorized_oracle_tv']-aa[(r['seed'],r['arm'])]['after']['factorized_oracle_tv']) for r in b if r['treatment']=='ema_recovery'))
print('initial hashes matched all stage1',all(len(set(r['initial_state_sha256'] for r in a if r['seed']==s))==1 for s in range(13,18)))
