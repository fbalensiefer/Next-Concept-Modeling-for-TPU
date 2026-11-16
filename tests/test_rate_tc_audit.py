import numpy as np
from ncm.rate_tc_audit import fit_models,evaluate,bootstrap_ci,evaluate_calibration_gates

def test_evaluation_is_out_of_sample_and_finite_at_66():
 r=np.random.default_rng(2);x=r.integers(0,2,(300,4));m=fit_models(x[:234],np.zeros((234,4,1),int));e=evaluate(m,x[234:],np.zeros((66,4,1),int))
 assert len(e['gain'])==66 and np.isfinite(e['tree']).all()
 assert len(bootstrap_ci(e['gain']))==2

def test_disjoint_identity_contexts_allow_conditional_independence():
 r=np.random.default_rng(3);ids=np.arange(1066);ctx=np.broadcast_to((ids%8)[:,None,None],(1066,4,1));p=(.1+.1*(ids%8)/8)[:,None];x=(r.random((1066,4))<p).astype(int)
 assert not set(ids[:1000])&set(ids[1000:])
 m=fit_models(x[:1000],ctx[:1000]);g=evaluate(m,x[1000:],ctx[1000:])['gain'].mean()
 assert g<.02

def test_parity_is_missed_by_pairwise_tree_and_seen_by_ar():
 r=np.random.default_rng(4);x=r.integers(0,2,(10000,3));x[:,2]=x[:,0]^x[:,1];c=np.zeros((len(x),3,1),int);m=fit_models(x[:5000],c[:5000]);e=evaluate(m,x[5000:],c[5000:])
 assert abs(e['gain'].mean())<.02
 assert (e['ind']-e['ar']).mean()>.1

def test_replication_runner_reports_accuracy_and_bootstrap_coverage():
 r=evaluate_calibration_gates(replications=2,eval_sizes=(12,),bootstrap_reps=20,seed=31)
 assert r['replications_per_cell']==2
 for law in ('markov','blocks'):
  cell=r['results'][law]['12']
  assert cell['replications']==2 and len(cell['replicate_gains'])==2
  assert cell['numerical_pass_fraction']==cell['numerical_passes']/2
  assert cell['empirical_coverage']==cell['coverage_count']/2
  assert cell['gate_pass']==(cell['numerical_pass_fraction']>=.95 and cell['empirical_coverage']>=.90)
