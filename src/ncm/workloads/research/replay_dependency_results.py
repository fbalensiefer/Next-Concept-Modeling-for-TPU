"""Verify all 480 committed prior fits using compact tables, without checkpoints."""
import hashlib,json
from pathlib import Path
import numpy as np
from ncm.dependence import codewords
from ncm.dependency_validation import MODELS,fit_law

folder=Path('reports/dependency-validation');meta=json.loads((folder/'codec_tables.json').read_text());path=folder/'codec_tables.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
data=np.load(path,allow_pickle=False);records=json.loads((folder/'results.json').read_text());lookup={(r['domain'],r['seed'],r['arm'],r['model']):r for r in records};verified=0
for item in meta['models']:
    domain=item['domain'];prefix=item['prefix'];tokens=data[domain+'_tokens'];p=data[domain+'_probability'];train=data[domain+'_train'];valid=data[domain+'_validation'];codes=data[prefix+'_codes'];dest=data[prefix+'_destinations'];reconstructed=data[prefix+'_reconstructed'];words=codewords((1,tokens.shape[1]),2);ids=codes@2**np.arange(codes.shape[1]-1,-1,-1);counts=np.bincount(train,minlength=len(tokens));cells={}
    for i,code in enumerate(ids):cells[code]=max(cells.get(code,0),int(counts[i]))
    ceiling=sum(cells.values())/len(train);accuracy=float(np.mean(np.all(reconstructed[valid]==tokens[valid],axis=1)));eligible=ceiling>=.98 and accuracy>=.98
    code_p=np.bincount(ids,weights=p,minlength=len(words));code_p/=code_p.sum()
    for model in MODELS:
        row=lookup[(domain,item['seed'],item['arm'],model)];q,complexity=fit_law(codes[train],words,model);pushed=np.bincount(dest,weights=q,minlength=len(tokens)+1);positive=code_p>0
        metrics=dict(output_tv=float(abs(pushed-np.append(p,0)).sum()/2),code_kl=float(np.sum(code_p[positive]*np.log(code_p[positive]/q[positive]))),validation_code_nll=float(-np.log(q[ids[valid]]).mean()),missing_source_mass=float(p[pushed[:-1]==0].sum()),outside_mass=float(pushed[-1]),training_collision_ceiling=ceiling,validation_sequence_accuracy=accuracy)
        for key,value in metrics.items():np.testing.assert_allclose(value,row[key],atol=1e-12,rtol=1e-12)
        assert row['codec_eligible']==eligible and row['complexity']==complexity and row['codec_sha256']==item['codec_sha256']
        verified+=1
assert verified==480
print('Verified all 480 fits, eligibility checks, conditional structures and checkpoint identities from committed compact tables.')
