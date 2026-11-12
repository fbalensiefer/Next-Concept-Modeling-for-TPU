import json,shutil
from pathlib import Path
import numpy as np
out=Path('reports/maintenance-ablation');out.mkdir(exist_ok=True)
allrows=json.loads((out/'results.json').read_text())
summary={}
for name in ['minimal_ema','ema_recovery','reference','fixed_binary']:
 rr=[r for r in allrows if r['treatment']==name and (name!='ema_recovery' or r['study']=='maintenance-ablation')]
 summary[name]=dict(models=len(rr),mean_sequence_reconstruction=float(np.mean([r['after']['sequence_reconstruction'] for r in rr])),minimum_sequence_reconstruction=min(r['after']['sequence_reconstruction'] for r in rr),mean_factorized_oracle_tv=float(np.mean([r['after']['factorized_oracle_tv'] for r in rr])),mean_factorized_oracle_kl_nats=float(np.mean([r['after']['factorized_oracle_kl_nats'] for r in rr])),population_collapsed_models=sum(r['after']['active_coarse_codes']<2 for r in rr),monitoring_failures=sum(r['collapsed_fraction']>.05 for r in rr),collision_models=sum(r['after']['population_unique_encodings']<16 for r in rr),missing_support_models=sum(r['audited_after']['factorized_missing_teacher_mass']>0 for r in rr),all_fidelity_usage_gates_passed=all(r['after']['sequence_reconstruction']>=.98 and r['after']['population_unique_encodings']==16 and r['audited_after']['factorized_missing_teacher_mass']==0 and r['after']['coarse_perplexity']>=1.5 and r['collapsed_fraction']<=.05 for r in rr),codec_step_flops=sorted(set(r['codec_step_flops'] for r in rr)))
# Seeds, rather than objective arms, are the paired replication units.
paired={}
for left,right in [('ema_recovery','minimal_ema'),('ema_recovery','reference'),('fixed_binary','ema_recovery')]:
 vals=[]
 for seed in range(13,18):
  def value(name):return np.mean([r['after']['factorized_oracle_tv'] for r in allrows if r['seed']==seed and r['treatment']==name and (name!='ema_recovery' or r['study']=='maintenance-ablation')])
  vals.append(float(value(left)-value(right)))
 paired[left+' minus '+right]=dict(seed_mean_tv_differences=vals,mean=float(np.mean(vals)),range=[min(vals),max(vals)])
(out/'analysis.json').write_text(json.dumps(dict(summary=summary,paired_seed_analysis=paired),indent=2)+'\n')
print(json.dumps(dict(summary=summary,paired=paired),indent=2))
