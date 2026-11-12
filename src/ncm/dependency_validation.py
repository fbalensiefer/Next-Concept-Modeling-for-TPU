"""Training-only tabular priors on frozen reference codecs; no production changes."""
import hashlib
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import serialization

from .config import experiment_rvq_config
from .dependence import codewords
from .grids import align, grid_matrices
from .rvq import CodebookState
from .scale_experiment import as_batch
from .tokenizer import MultiscaleTokenizer

MODELS = ('independent', 'adjacent_pairs', 'conditional_tree', 'joint_fine')


def tree_parents(samples):
    """Maximum spanning tree of empirical conditional MI; center-rooted."""
    length=samples.shape[1]-1
    weights=[]
    for i in range(1,length+1):
        for j in range(i+1,length+1):
            value=0.
            for z in (0,1):
                subset=samples[samples[:,0]==z]
                if not len(subset):continue
                p=np.bincount(2*subset[:,i]+subset[:,j],minlength=4).reshape(2,2)/len(subset)
                product=p.sum(1)[:,None]*p.sum(0)[None,:];positive=p>0
                value+=len(subset)/len(samples)*np.sum(p[positive]*np.log(p[positive]/product[positive]))
            weights.append((-float(value),i,j))
    groups=list(range(length+1));edges={i:[] for i in range(1,length+1)}
    def find(i):
        while groups[i]!=i:i=groups[i]
        return i
    for _,i,j in sorted(weights):
        a,b=find(i),find(j)
        if a!=b:groups[a]=b;edges[i].append(j);edges[j].append(i)
    def distances(root):
        d={root:0};queue=[root]
        for node in queue:
            for child in edges[node]:
                if child not in d:d[child]=d[node]+1;queue.append(child)
        return d
    root=min(edges,key=lambda i:(max(distances(i).values()),i))
    parents={root:None};queue=[root];depth={root:0}
    for node in queue:
        for child in sorted(edges[node]):
            if child not in parents:parents[child]=node;depth[child]=depth[node]+1;queue.append(child)
    return parents,2+max(depth.values())


def fit_law(samples, words, model, prior_mass=1.):
    """Posterior predictive tables: total symmetric pseudo-count per row is one.

    Only observed training codes enter this function. Geometry is one coarse
    bit followed by length fine bits. Trees use training conditional MI only.
    """
    if model not in MODELS:raise ValueError(model)
    samples=np.asarray(samples,np.int32);words=np.asarray(words,np.int32)
    if samples.ndim!=2 or not len(samples) or samples.shape[1]!=words.shape[1]:raise ValueError('Invalid code samples')
    length=words.shape[1]-1
    if model=='independent':blocks=[([i],[0]) for i in range(1,length+1)];rounds=2
    elif model=='adjacent_pairs':blocks=[(list(range(i,min(i+2,length+1))),[0]) for i in range(1,length+1,2)];rounds=2
    elif model=='joint_fine':blocks=[(list(range(1,length+1)),[0])];rounds=2
    else:
        parents,rounds=tree_parents(samples)
        blocks=[([node],[0] if parent is None else [0,parent]) for node,parent in parents.items()]
    def encode(array,columns):return array[:,columns]@2**np.arange(len(columns)-1,-1,-1)
    q=np.ones(len(words));root=np.bincount(samples[:,0],minlength=2)
    q*=(root[words[:,0]]+prior_mass/2)/(len(samples)+prior_mass)
    parameters=1
    for targets,context in blocks:
        categories=2**len(targets);contexts=2**len(context)
        counts=np.zeros((contexts,categories));np.add.at(counts,(encode(samples,context),encode(samples,targets)),1)
        probabilities=(counts+prior_mass/categories)/(counts.sum(-1,keepdims=True)+prior_mass)
        q*=probabilities[encode(words,context),encode(words,targets)]
        parameters+=contexts*(categories-1)
    assert np.all(q>0) and np.isclose(q.sum(),1.,atol=1e-12)
    return q,dict(free_parameters=parameters,serial_rounds=rounds,
                  conditionals=[dict(targets=t,context=c) for t,c in blocks])


def codec_tables(tokenizer, state, tokens):
    cfg=tokenizer.config;books=CodebookState(**{k:jnp.asarray(v) for k,v in state['codebooks'].items()})
    batch=as_batch(tokens);o=tokenizer.apply({'params':state['params']},batch.tokens,batch.mask,batch.fallback_bytes,batch.fallback_mask,books)
    codes=np.concatenate([np.asarray(o.rvq.indices[k,:,:n]) for k,n in enumerate(cfg.lengths)],axis=1)
    words=codewords(cfg.lengths,2);_,up=grid_matrices(cfg);representation=jnp.zeros((len(words),cfg.max_length,cfg.dim));offset=0
    for k,n in enumerate(cfg.lengths):
        vectors=books.embeddings[k][words[:,offset:offset+n]];vectors=jnp.pad(vectors,((0,0),(0,cfg.max_length-n),(0,0)))
        valid=jnp.broadcast_to(jnp.arange(cfg.max_length)<n,(len(words),cfg.max_length));representation+=align(vectors,valid,up[k])[0];offset+=n
    decoded=np.asarray(tokenizer.apply({'params':state['params']},representation,jnp.ones(representation.shape[:2],bool),method=tokenizer.decode).argmax(-1))
    valid=np.all((decoded>=65)&(decoded<=66),axis=1)
    destinations=np.where(valid,(decoded-65)@2**np.arange(cfg.max_length-1,-1,-1),len(tokens))
    return codes,words,destinations,np.asarray(o.logits.argmax(-1))


def run():
    root=Path('runs/fresh-domain-validation');output=Path('runs/dependency-validation');output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('Existing experiment; use a fresh directory')
    rows=json.loads((root/'results.json').read_text());reference=[r for r in rows if r['treatment']=='reference'];assert len(reference)==120
    original=json.loads((root/'protocol.json').read_text())
    protocol=dict(models=MODELS,scope='All 120 frozen reference codecs; every objective arm retained; no neural retraining or production changes.',
        fit='512 empirical training code observations only. Total symmetric Dirichlet pseudo-count one per conditional row. Conditional tree topology uses training conditional MI and deterministic tie-breaking; tree center minimizes serial depth.',
        fidelity_gate='Flag validation sequence accuracy <.98 or training collision ceiling <.98. No population probabilities used for this gate. Gate is a diagnostic eligibility check, not a population certificate; retain and report all codecs.',
        evaluation='Exact source output TV, KL, support/outside mass, empirical validation code NLL, and serial depth/parameter count. Population used only after fitting; no smoothing/model selection on evaluation results.',
        source_results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numeric='NumPy float64 tables; JAX used only for unchanged frozen codec inference.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    for row in reference:
        domain=row['domain'];length=row['length'];cfg=dict(original['config'],lengths=[1,length]);tokenizer=MultiscaleTokenizer(experiment_rvq_config(cfg),256)
        d=np.load(root/domain/'domain.npz');tokens=d['tokens'];p=d['probability'];train=d['train'];validation=d['validation']
        checkpoint=root/domain/f"seed-{row['seed']}"/row['arm']/'reference/codec_repaired.msgpack'
        raw=checkpoint.read_bytes();digest=hashlib.sha256(raw).hexdigest();state=serialization.msgpack_restore(raw)
        codes,words,destinations,decoded=codec_tables(tokenizer,state,tokens)
        ids=codes@2**np.arange(codes.shape[1]-1,-1,-1)
        train_counts=np.bincount(train,minlength=len(tokens));cell={}
        for i,code in enumerate(ids):cell[code]=max(cell.get(code,0),int(train_counts[i]))
        training_ceiling=sum(cell.values())/len(train)
        validation_accuracy=float(np.mean(np.all(decoded[validation]==tokens[validation],axis=1)))
        eligible=training_ceiling>=.98 and validation_accuracy>=.98
        code_p=np.bincount(ids,weights=p,minlength=len(words));code_p/=code_p.sum()
        for model in MODELS:
            start=time.perf_counter();q,complexity=fit_law(codes[train],words,model);fit_seconds=time.perf_counter()-start
            pushed=np.bincount(destinations,weights=q,minlength=len(tokens)+1);positive=code_p>0
            missing=float(p[pushed[:-1]==0].sum());padded=np.append(p,0);selection=padded>0
            item=dict(domain=domain,seed=row['seed'],arm=row['arm'],model=model,
                codec_eligible=eligible,validation_sequence_accuracy=validation_accuracy,
                training_collision_ceiling=training_ceiling,population_codec_failures=row['failures'],
                output_tv=float(abs(pushed-padded).sum()/2),missing_source_mass=missing,outside_mass=float(pushed[-1]),
                output_forward_kl=None if missing else float(np.sum(padded[selection]*np.log(padded[selection]/pushed[selection]))),
                code_kl=float(np.sum(code_p[positive]*np.log(code_p[positive]/q[positive]))),
                validation_code_nll=float(-np.log(q[ids[validation]]).mean()),fit_seconds=fit_seconds,
                complexity=complexity,codec_sha256=digest)
            results.append(item)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(dict(domain=domain,seed=row['seed'],arm=row['arm'],eligible=eligible)),flush=True)
    assert len(results)==480
    print('COMPLETED 480',flush=True)


if __name__=='__main__':run()
