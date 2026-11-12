"""Replay alignment diagnostics from committed arrays; no checkpoint required."""
import hashlib,json
from pathlib import Path
import numpy as np


def diagnose(folder=Path('reports/alignment-diagnosis')):
    meta=json.loads((folder/'inputs.json').read_text());path=folder/'diagnostic_inputs.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    d=np.load(path,allow_pickle=False);p=d['probability'];target=d['canonical']
    def soft(logits):
        z=logits-logits.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)
    true=soft(d['teacher_hidden']@d['head_kernel']+d['head_bias'])
    def predictive_kl(h):
        restored=(h*d['scale'])@d['projection'].T+d['mean'];q=soft(restored@d['head_kernel']+d['head_bias']);return float(p@(true*(np.log(true)-np.log(np.maximum(q,1e-30)))).sum(-1).mean(-1))
    last_entropy=0.
    for i in range(0,len(p),2):
        pair=p[i:i+2];q=pair/pair.sum();last_entropy-=float(pair.sum()*np.sum(q*np.log2(q)))
    invariants=dict(first_position_target_max_range=float(np.ptp(target[:,0],axis=0).max()),last_bit_pair_target_max_difference=float(abs(target[::2]-target[1::2]).max()),last_bit_conditional_entropy_bits=last_entropy,pca_teacher_predictive_kl_nats=predictive_kl(target))
    rows=[]
    for index,model in enumerate(meta['models']):
        latent=d['latents'][index];quantized=d['quantized'][index]
        rows.append(dict(model,latent_mse=float(p@((latent-target)**2).mean((1,2))),quantized_mse=float(p@((quantized-target)**2).mean((1,2))),latent_predictive_kl=predictive_kl(latent),quantized_predictive_kl=predictive_kl(quantized)))
    means={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm])) for key in ('latent_mse','quantized_mse','latent_predictive_kl','quantized_predictive_kl')} for arm in dict.fromkeys(r['arm'] for r in rows)}
    return dict(scope='One historical teacher and finite domain; feature-head compatibility diagnostic, not a causal intervention.',invariants=invariants,means=means,models=rows)


if __name__=='__main__':
    result=diagnose();Path('reports/alignment-diagnosis/results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['invariants'],indent=2))
