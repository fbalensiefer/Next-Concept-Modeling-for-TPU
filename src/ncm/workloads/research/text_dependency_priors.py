"""Stationary iid versus sparse first-order priors on three representations.

Diagnostic CPU study only: table priors do not model position or cross-level
conditioning and sample quality is not certified by byte diagnostics.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
from ncm.prior_experiment import load_codec
from ncm.experiment import batch_at
from ncm.generation import encode
from ncm.grids import grid_matrices
from ncm.tokenizer import MultiscaleTokenizer
from ncm.text_data import document_dataset, validate_splits
from ncm.representations import encode_pairs, decode_pairs
from ncm.table_prior import TablePrior
from ncm.benchmark import quality_summary


def diagnostics(tokens, reference, train):
    result=quality_summary(tokens,reference)
    training={row.tobytes() for row in np.asarray(train,np.uint8)}
    result['exact_training_window_match_fraction']=float(np.mean([row.tobytes() in training for row in np.asarray(tokens,np.uint8)]))
    result['adjacent_equal_byte_fraction']=float(np.mean(tokens[:,1:]==tokens[:,:-1]))
    return result


def codec_functions(directory):
    model,state,cfg,digest=load_codec(directory)
    params=state.params['MultiscaleTokenizer_0']
    encoder=jax.jit(lambda tokens: encode(model,params,state.codebooks,
        batch_at((tokens,jnp.ones_like(tokens,bool)),jnp.arange(len(tokens)))).indices)
    _,up=grid_matrices(model.rvq)
    lengths=model.rvq.lengths
    embeddings=jnp.asarray(state.codebooks.embeddings)
    @jax.jit
    def decoder(codes):
        cumulative=jnp.zeros((len(codes),128,model.rvq.dim))
        offset=0
        for level,length in enumerate(lengths):
            vectors=embeddings[level][codes[:,offset:offset+length]]
            cumulative=cumulative+jnp.einsum('ln,bnd->bld',up[level,:,:length],vectors)
            offset+=length
        logits=MultiscaleTokenizer(model.rvq,256).apply({'params':params},cumulative,
            jnp.ones((len(codes),128),bool),method=MultiscaleTokenizer.decode)
        return logits.argmax(-1)
    return encoder,decoder,lengths,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='runs/real-text-pilot')
    parser.add_argument('--codecs',default='runs/reconstruction-capacity-v1')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise ValueError('Use a fresh output directory')
    paths=[Path(args.data)/f'{s}.jsonl' for s in ('train','validation')]
    validate_splits(paths)
    train,_=document_dataset(paths[0],128)
    valid,_=document_dataset(paths[1],128)
    codecs=[Path(args.codecs)/f'codes-64-seed-{s}' for s in (0,1,2)]
    hashes={str(p):hashlib.sha256((p/'checkpoint.msgpack').read_bytes()).hexdigest() for p in codecs}
    output.mkdir(parents=True)
    sources=[*Path('src/ncm').glob('*.py'),Path(__file__).resolve().relative_to(Path.cwd())]
    for p in sources:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    protocol=dict(data_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        codec_hashes=hashes, sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        alpha=1., sample_seeds=[0,1,2], samples_per_seed=256, timing_batch=16,
        timed_repeats=7, length=128, devices=[str(d) for d in jax.devices()],
        versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','flax','optax','numpy')},
        factorization='Stationary marginal or first-order within each sequence; RVQ levels independent',
        nll_rule='Compare iid versus Markov only within the same representation and codec',
        selection='All 64-entry codecs retained; no quality or timing selection; no test evaluation',
        limitations='One training and validation book; stationary table controls; no quality-matched neural AR or TPU comparison')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    for name,directory in [('raw_bytes',None),('fixed_pairs',None)]+[(f'rvq_seed_{s}',p) for s,p in enumerate(codecs)]:
        encode_seconds=0.
        if directory is None:
            arrays=[train[0] if name=='raw_bytes' else encode_pairs(train[0]),
                    valid[0] if name=='raw_bytes' else encode_pairs(valid[0])]
            lengths=[arrays[0].shape[1]];vocabulary=256 if name=='raw_bytes' else 65536
            decoder=lambda x: x if name=='raw_bytes' else decode_pairs(x)
        else:
            encoder,decoder,lengths,digest=codec_functions(directory)
            started=time.perf_counter();arrays=[]
            for data in (train,valid):
                chunks=[]
                for start in range(0,len(data[0]),64):
                    encoded=np.asarray(encoder(jnp.asarray(data[0][start:start+64])))
                    chunks.append(np.concatenate([encoded[k,:,:n] for k,n in enumerate(lengths)],axis=1))
                arrays.append(np.concatenate(chunks))
            encode_seconds=time.perf_counter()-started;vocabulary=64
        # Decoder identity is audited against recorded autoencoding accuracy.
        reconstructed=np.concatenate([np.asarray(decoder(arrays[1][i:i+16])) for i in range(0,len(arrays[1]),16)])
        reconstruction=float(np.mean(reconstructed==valid[0]))
        if directory is not None:
            expected=json.loads((directory/'metrics.jsonl').read_text().splitlines()[-1])['validation']['byte_accuracy']
            if abs(expected-reconstruction)>1e-10:raise ValueError('Code decode differs from frozen codec audit')
        for structured in (False,True):
            started=time.perf_counter();priors=[];offset=0;heldout=[]
            for length in lengths:
                prior=TablePrior(arrays[0][:,offset:offset+length],vocabulary,structured)
                priors.append(prior);heldout.append(prior.nll(arrays[1][:,offset:offset+length]));offset+=length
            fit_seconds=time.perf_counter()-started
            def generate(rng,count):
                codes=np.concatenate([p.sample(rng,count,n) for p,n in zip(priors,lengths)],axis=1)
                return np.asarray(decoder(codes))
            # Warmup includes device compilation/transfer and is reported separately.
            started=time.perf_counter();generate(np.random.default_rng(999),16)
            warmup_seconds=time.perf_counter()-started
            times=[]
            for repeat in range(7):
                started=time.perf_counter();generate(np.random.default_rng(1000+repeat),16)
                times.append(time.perf_counter()-started)
            samples=[]
            for seed in (0,1,2):
                tokens=generate(np.random.default_rng(seed),256)
                samples.append(dict(seed=seed,diagnostics=diagnostics(tokens,valid,train[0])))
            results.append(dict(representation=name,prior='markov' if structured else 'independent',
                vocabulary=vocabulary,sequence_lengths=lengths,
                validation_nll_per_code=float(np.average(heldout,weights=lengths)),
                validation_nll_per_sequence=float(np.dot(heldout,lengths)),level_nll=heldout,
                validation_reconstruction_byte_accuracy=reconstruction,
                encode_wall_seconds=encode_seconds,fit_wall_seconds=fit_seconds,
                warmup_seconds=warmup_seconds,generation_seconds_batch16=times,
                numeric_prior_payload_bytes=sum(p.storage_bytes() for p in priors),samples=samples))
            (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    for p,digest in hashes.items():
        if hashlib.sha256((Path(p)/'checkpoint.msgpack').read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen checkpoint changed')
    (output/'integrity.json').write_text(json.dumps(dict(frozen_checkpoints_unchanged=True,completed_fits=len(results)))+'\n')


if __name__=='__main__':main()
