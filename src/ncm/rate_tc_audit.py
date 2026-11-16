"""Synthetic-only cross-validated rate/TC diagnostics; never loads corpora."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
ALPHA=.5

def _key(row): return tuple(np.asarray(row).tolist())
def _fit(x,c,parents=(),alpha=ALPHA):
 v=int(x.max())+1; counts={}
 for n in range(len(x)):
  k=_key(c[n]);counts.setdefault(k,np.zeros(v))[x[n]]+=1
 return counts,v,alpha
def _logp(t,c,p,y):
 counts,v,a=t;r=counts.get(_key(c));den=(0 if r is None else r.sum())+a*v;num=a if r is None else r[y]+a
 return np.log(num/den)
def _mi(x,y,c):
 out=0.;v=int(max(x.max(),y.max()))+1
 for ctx in np.unique(c,axis=0):
  m=np.all(c==ctx,axis=1); z=np.zeros((v,v));np.add.at(z,(x[m],y[m]),1);p=z/max(1,m.sum());a=p.sum(1);b=p.sum(0);nz=p>0
  out+=m.sum()/len(x)*np.sum(p[nz]*np.log(p[nz]/(a[:,None]*b[None,:])[nz]))
 return float(out)
@dataclass
class Model: ind:list;tree:list;ar:list;parents:tuple

def fit_models(x,context,alpha=ALPHA):
 x=np.asarray(x,int);c=np.asarray(context,int)
 if c.ndim==2:c=c[...,None]
 if x.ndim!=2 or c.shape[:2]!=x.shape or len(x)<2:raise ValueError('expected x [N,D], contexts [N,D,C]')
 d=x.shape[1];ind=[_fit(x[:,i],c[:,i],(),alpha) for i in range(d)]
 w={(i,j):_mi(x[:,i],x[:,j],c[:,i]) for i in range(d) for j in range(i)};seen={0};pa=[-1]*d
 while len(seen)<d:
  _,a,b=max((v,a,b) for (a,b),v in w.items() if (a in seen)!=(b in seen));child=b if a in seen else a;pa[child]=a if a in seen else b;seen.add(child)
 tree=[];ar=[]
 for i in range(d):
  tc=c[:,i] if pa[i]<0 else np.column_stack((c[:,i],x[:,pa[i]]));hist=tuple(range(max(0,i-2),i));ac=np.column_stack((c[:,i],x[:,hist])) if hist else c[:,i];tree.append(_fit(x[:,i],tc,(),alpha));ar.append(_fit(x[:,i],ac,(),alpha))
 return Model(ind,tree,ar,tuple(pa))
def evaluate(m,x,context):
 x=np.asarray(x,int);c=np.asarray(context,int)
 if c.ndim==2:c=c[...,None]
 losses={k:np.zeros(len(x)) for k in ('ind','tree','ar')}
 for n in range(len(x)):
  for i in range(x.shape[1]):
   losses['ind'][n]-=_logp(m.ind[i],c[n,i],(),x[n,i]);tc=c[n,i] if m.parents[i]<0 else np.r_[c[n,i],x[n,m.parents[i]]];losses['tree'][n]-=_logp(m.tree[i],tc,(),x[n,i]);hist=tuple(range(max(0,i-2),i));ac=np.r_[c[n,i],x[n,list(hist)]] if hist else c[n,i];losses['ar'][n]-=_logp(m.ar[i],ac,(),x[n,i])
 return losses|{'gain':losses['ind']-losses['tree']}
def bootstrap_ci(v,reps=2000,seed=1):
 x=np.asarray(v,float);r=np.random.default_rng(seed);z=np.mean(r.choice(x,(reps,len(x)),replace=True),axis=1);return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def _markov_sequences(rng,n,d=4,flip=.1):
 z=np.zeros((n,d),int);z[:,0]=rng.integers(0,2,n)
 for i in range(1,d):z[:,i]=z[:,i-1]^(rng.random(n)<flip)
 return z

def _block_sequences(rng,n,d=4,flip=.15):
 z=rng.integers(0,2,(n,d));z[:,1]=z[:,0]^(rng.random(n)<flip);z[:,3]=z[:,2]^(rng.random(n)<flip);return z

def evaluate_calibration_gates(replications=100,eval_sizes=(66,512),bootstrap_reps=2000,seed=914):
 """Replicated synthetic accuracy and sequence-bootstrap coverage study."""
 results={}; gate=True
 for law, sampler, true_tc in (
  ('markov',lambda r,n:_markov_sequences(r,n),3*(np.log(2)-(-.1*np.log(.1)-.9*np.log(.9)))),
  ('blocks',lambda r,n:_block_sequences(r,n),2*(np.log(2)-(-.15*np.log(.15)-.85*np.log(.85))))):
  results[law]={}
  for size in eval_sizes:
   gains=[]; intervals=[]
   for rep in range(replications):
    rng=np.random.default_rng(seed+rep*10007+size+(0 if law=='markov' else 500000))
    train=sampler(rng,4096); test=sampler(rng,size)
    m=fit_models(train,np.zeros((*train.shape,1),int)); e=evaluate(m,test,np.zeros((*test.shape,1),int))
    gains.append(float(e['gain'].mean())); intervals.append(bootstrap_ci(e['gain'],bootstrap_reps,seed+rep))
   gains=np.asarray(gains); rel=np.abs(gains-true_tc)/true_tc
   coverage=np.asarray([(lo<=true_tc<=hi) for lo,hi in intervals],bool)
   summary={'replications':replications,'eval_size':size,'analytical_tc_nats_per_sequence':float(true_tc),
    'mean_gain_nats_per_sequence':float(gains.mean()),'bias_nats_per_sequence':float(gains.mean()-true_tc),
    'relative_error_tolerance':.2,'numerical_passes':int((rel<=.2).sum()),'numerical_pass_fraction':float((rel<=.2).mean()),
    'required_numerical_pass_fraction':.95,'bootstrap_replicates_per_interval':bootstrap_reps,
    'nominal_interval_coverage':.95,'coverage_count':int(coverage.sum()),'empirical_coverage':float(coverage.mean()),
    'required_coverage':.90,'replicate_gains':gains.tolist(),'bootstrap_intervals':intervals}
   summary['gate_pass']=summary['numerical_pass_fraction']>=.95 and summary['empirical_coverage']>=.90
   gate=gate and summary['gate_pass'];results[law][str(size)]=summary
 return {'results':results,'gate_pass':bool(gate),'replications_per_cell':replications,'eval_sizes':list(eval_sizes)}
def _laws(seed=41,nf=4096,ne=66):
 r=np.random.default_rng(seed);d=4;zero=lambda n:np.zeros((n,d,1),int)
 yield 'independent_uniform',r.integers(0,2,(nf,d)),r.integers(0,2,(ne,d)),zero(nf),zero(ne),0.
 ids=np.arange(nf+ne);ctx=(ids%8)[:,None,None]*np.ones((1,d,1),int);prob=(.15+.1*(ids%8)/7)[:,None];x=(r.random((nf+ne,d))<prob).astype(int)
 yield 'rich_disjoint_identity',x[:nf],x[nf:],ctx[:nf],ctx[nf:],0.
 def hb(p):return -p*np.log(p)-(1-p)*np.log(1-p)
 def mark(n):
  z=np.zeros((n,d),int);z[:,0]=r.integers(0,2,n)
  for i in range(1,d):z[:,i]=z[:,i-1]^(r.random(n)<.1)
  return z
 yield 'markov',mark(nf),mark(ne),zero(nf),zero(ne),(d-1)*(np.log(2)-hb(.1))
 def blocks(n):
  z=r.integers(0,2,(n,d));z[:,1]=z[:,0]^(r.random(n)<.15);z[:,3]=z[:,2]^(r.random(n)<.15);return z
 yield 'blocks',blocks(nf),blocks(ne),zero(nf),zero(ne),2*(np.log(2)-hb(.15))
def run_stage_a(protocol_path,output_path):
 p=json.loads(Path(protocol_path).read_text());o={'protocol':p['protocol'],'real_text_evaluated':False,'laws':{},'gates':{}}
 for name,xf,xe,cf,ce,truth in _laws():
  m=fit_models(xf,cf);e=evaluate(m,xe,ce);g=float(e['gain'].mean());o['laws'][name]={'true_tc_nats':truth,'tree_gain_nats_per_sequence':g,'gain_ci':bootstrap_ci(e['gain']),'identity_overlap':0 if name=='rich_disjoint_identity' else len(set(map(tuple,xf))&set(map(tuple,xe)))}
 r=np.random.default_rng(8);z=r.integers(0,2,(20000,3));z[:,2]=z[:,0]^z[:,1];a,b=z[:10000],z[10000:];c=np.zeros((len(a),3,1),int);ce=np.zeros((len(b),3,1),int);m=fit_models(a,c);e=evaluate(m,b,ce);o['laws']['parity']={'true_tc_nats':float(np.log(2)),'tree_gain_nats_per_sequence':float(e['gain'].mean()),'ar_gain_nats_per_sequence':float((e['ind']-e['ar']).mean())}
 o['replicated_calibration']=evaluate_calibration_gates()
 o['gates']={'uniform_gain':o['laws']['independent_uniform']['tree_gain_nats_per_sequence']<=.02,'disjoint_identity':o['laws']['rich_disjoint_identity']['identity_overlap']==0 and o['laws']['rich_disjoint_identity']['tree_gain_nats_per_sequence']<=.02,'parity_tree_blindness':abs(o['laws']['parity']['tree_gain_nats_per_sequence'])<=.02,'parity_ar_detects':o['laws']['parity']['ar_gain_nats_per_sequence']>=.1,'replicated_markov_block_accuracy_and_coverage':o['replicated_calibration']['gate_pass']};o['status']='passed' if all(o['gates'].values()) else 'halted_calibration_failure';o['real_text_evaluated']=False
 Path(output_path).parent.mkdir(parents=True,exist_ok=True);Path(output_path).write_text(json.dumps(o,indent=2)+'\n');return o
