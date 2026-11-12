"""Fresh-seed validation on prespecified exact binary source laws.

python -m ncm.domain_validation
The neural teacher supplies alignment features; the analytic source law remains
both the sampling and evaluation target, including its higher-order dependence.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from .autoregressive import AutoregressiveBytes, shifted_inputs
from .coarse import maintain_coarse, optimizer_state, warmup_codebooks
from .config import TransformerConfig, experiment_rvq_config
from .coverage import FrozenCodec, codec_training_distribution, make_decoder_repair_step
from .dependence import codewords
from .generation import PriorState
from .losses import cross_entropy
from .maintenance_experiment import ema_recovery, exact_metrics, fixed_binary
from .rvq import init_codebooks
from .scale_experiment import ALL_ARMS, arm_weights, as_batch, fit_canonical, make_codec_step, save_state, state_bytes
from .tokenizer import MultiscaleTokenizer
from .training import TrainState

SEEDS = (31, 32, 33, 34, 35)
DOMAINS = (
    ('original_mixture', 4, 9101),
    ('biased_independent', 4, 9102),
    ('persistent_markov', 8, 9103),
    ('noisy_parity', 8, 9104),
    ('skewed_modes', 8, 9105),
    ('alternating_modes', 8, 9106),
)
TREATMENTS = {'ema_recovery': ema_recovery, 'reference': maintain_coarse,
              'fixed_binary': fixed_binary}


def source_law(name, length):
    bits = codewords((length,), 2)
    uniform = np.full(len(bits), 1 / len(bits), np.float64)
    if name == 'original_mixture':
        p = .25 * uniform
        p[0] += .375
        p[-1] += .375
    elif name == 'biased_independent':
        p = np.prod(np.where(bits == 1, .2, .8), axis=1)
    elif name == 'persistent_markov':
        p = .5 * np.prod(np.where(bits[:, 1:] == bits[:, :-1], .85, .15), axis=1)
    elif name == 'noisy_parity':
        p = .1 * uniform + .9 * (bits.sum(-1) % 2 == 0) / (len(bits) / 2)
    elif name == 'skewed_modes':
        p = .1 * uniform
        p[0] += .8
        p[-1] += .1
    elif name == 'alternating_modes':
        p = .2 * uniform
        alternating = np.arange(length) % 2
        p[np.all(bits == alternating, axis=1)] += .4
        p[np.all(bits == 1-alternating, axis=1)] += .4
    else:
        raise ValueError(name)
    assert np.all(p > 0) and np.isclose(p.sum(), 1)
    return bits + 65, p


def prepare_teacher(config, tokens, p, seed, directory):
    model = AutoregressiveBytes(TransformerConfig(config['teacher_width'], 2, 1), tokens.shape[1])
    batch = as_batch(tokens)
    params = model.init(jax.random.key(seed), shifted_inputs(batch.tokens), batch.mask)['params']
    optimizer = optax.adam(config['teacher_learning_rate'])
    state = PriorState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))

    def step(state, batch):
        def objective(params):
            logits = model.apply({'params': params}, shifted_inputs(batch.tokens), batch.mask)[..., 65:67]
            return cross_entropy(logits, batch.tokens-65, batch.mask)
        loss, grads = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        return PriorState(optax.apply_updates(state.params, updates), opt_state, state.step+1), loss

    batches = [as_batch(tokens[np.random.default_rng(np.random.SeedSequence([seed, i, 10])).choice(
        len(tokens), config['batch_size'], p=p)]) for i in range(config['teacher_steps'])]
    jax.block_until_ready((state, batches))
    start = time.perf_counter()
    executable = jax.jit(step).lower(state, batches[0]).compile()
    compile_seconds = time.perf_counter()-start
    jax.block_until_ready(executable(state, batches[0]))
    start = time.perf_counter()
    for b in batches:
        state, loss = executable(state, b)
    jax.block_until_ready((state, loss))
    seconds = time.perf_counter()-start
    save_state(directory/'teacher.msgpack', state)
    hidden = np.asarray(model.apply({'params': state.params}, shifted_inputs(batch.tokens),
                                   batch.mask, method=model.represent))
    logits = model.apply({'params': state.params}, shifted_inputs(batch.tokens), batch.mask)[..., 65:67]
    logp = np.asarray(jax.nn.log_softmax(logits), np.float64)
    teacher_p = np.exp(np.take_along_axis(logp, (tokens-65)[..., None], -1)[..., 0].sum(-1))
    assert np.isclose(teacher_p.sum(), 1, atol=3e-6)
    teacher_p /= teacher_p.sum()
    draws = {name: np.random.default_rng(np.random.SeedSequence([seed, stream])).choice(
        len(tokens), config[field], p=p) for name, field, stream in (
            ('train', 'train_samples', 20), ('validation', 'validation_samples', 21),
            ('calibration', 'calibration_samples', 22))}
    mean, projection, scale = fit_canonical(hidden[draws['calibration']], config['dim'])
    canonical = ((hidden-mean) @ projection) / scale
    np.savez(directory/'domain.npz', tokens=tokens, probability=p, teacher_probability=teacher_p,
             canonical=canonical, mean=mean, projection=projection, scale=scale, **draws)
    info = dict(teacher_seed=seed, source_to_teacher_kl=float(np.sum(p*np.log(p/teacher_p))),
                source_to_teacher_tv=float(abs(p-teacher_p).sum()/2), teacher_seconds=seconds,
                teacher_compile_seconds=compile_seconds, observed_training_sequences=len(np.unique(draws['train'])),
                domain_sha256=hashlib.sha256((directory/'domain.npz').read_bytes()).hexdigest())
    (directory/'teacher_info.json').write_text(json.dumps(info, indent=2)+'\n')
    return canonical, draws, info


def run(resume=False):
    output = Path('runs/fresh-domain-validation')
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/'protocol.json'
    if protocol_path.exists() and not resume:
        raise ValueError('Existing protocol; use --resume or a fresh output directory')
    if resume and not protocol_path.exists():
        raise ValueError('No protocol to resume')
    config = json.loads(Path('examples/scale/coarse_stability.json').read_text())
    # Five fresh seeds; data quantities and training budgets stay unchanged.
    config.update(seeds=list(SEEDS), confirmation_seeds=list(SEEDS))
    protocol = dict(config=config, domains=DOMAINS, treatments=list(TREATMENTS),
        target='Analytic full-support source law, not fitted teacher law; independently trained teachers provide PCA alignment features only.',
        budgets='360 models, each 400 codec + 200 decoder replay updates; six teachers x 400 updates. No prior training.',
        fidelity_checks=dict(sequence_reconstruction_min=.98, missing_source_mass_max=.01,
                             outside_domain_mass_max=.01, collision_ceiling_min=.98),
        usage='Report final active codes, perplexity and collapsed-window fraction. Do not require balanced coarse usage for a skewed source.',
        comparison='Per-domain, seed-paired means across objective arms; TV difference <=.02 is a descriptive tolerance, not a statistical noninferiority claim. Report all failures, no tuning or domain exclusion.',
        limitations='Finite binary mechanism studies; not natural-language, unseen-support, pretrained transfer, or TPU validation. One feature teacher per source.',
        timing='Synchronized training windows, precomputed resident batches, discarded warmup, separate compilation and replay. No concurrent tests scheduled.',
        versions=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__),
        backend=jax.default_backend(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    if not resume:
        protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for p in Path(__file__).parent.glob('*.py'):
            (snapshot/p.name).write_bytes(p.read_bytes())
    else:
        frozen=json.loads(protocol_path.read_text())
        for field in ('config','treatments','target','budgets','fidelity_checks','usage','comparison'):
            if frozen[field]!=protocol[field]:
                raise ValueError('Changed experimental protocol: '+field)
    compiled = {}
    rows=[]
    teacher_infos={}
    optimizer=optax.adam(config['codec_learning_rate'])
    repair_optimizer=optax.adam(config['decoder_repair_learning_rate'])
    repair_fn=jax.jit(make_decoder_repair_step(config['dim'], repair_optimizer))
    for di, (name, length, teacher_seed) in enumerate(DOMAINS):
        directory=output/name;directory.mkdir(exist_ok=resume)
        cfg=dict(config, lengths=[1,length])
        rvq=experiment_rvq_config(cfg);tokenizer=MultiscaleTokenizer(rvq,256)
        tokens,p=source_law(name,length)
        if resume and (directory/'domain.npz').exists():
            loaded=np.load(directory/'domain.npz')
            np.testing.assert_array_equal(tokens,loaded['tokens'])
            np.testing.assert_array_equal(p,loaded['probability'])
            canonical=loaded['canonical'];draws={key:loaded[key] for key in ('train','validation','calibration')}
            teacher_info=json.loads((directory/'teacher_info.json').read_text())
        else:
            canonical,draws,teacher_info=prepare_teacher(cfg,tokens,p,teacher_seed,directory)
        teacher_infos[name]=teacher_info
        candidates,importance=codec_training_distribution(draws['train'],len(tokens),cfg['codec_sampling'])
        if length not in compiled:
            compiled[length]={n:jax.jit(make_codec_step(cfg,optimizer,maintenance=fn)) for n,fn in TREATMENTS.items()}
        for seed in SEEDS:
            key=jax.random.key(seed);books=init_codebooks(jax.random.fold_in(key,1),rvq)
            first=as_batch(tokens[draws['train'][np.arange(cfg['batch_size'])%len(draws['train'])]])
            params=tokenizer.init(jax.random.fold_in(key,2),first.tokens,first.mask,
                                  first.fallback_bytes,first.fallback_mask,books)['params']
            optstate=optimizer_state(optimizer.init(params),books,rvq,jax.random.fold_in(key,4))
            start=time.perf_counter()
            warmups=[as_batch(tokens[np.random.default_rng(np.random.SeedSequence([seed,i,30])).choice(
                candidates,cfg['batch_size'])]) for i in range(rvq.coarse.warmup_batches)]
            books,coarse=warmup_codebooks(tokenizer,params,books,optstate['coarse'],warmups)
            initial=TrainState(params,dict(optstate,coarse=coarse),books,jnp.asarray(0,jnp.int32))
            jax.block_until_ready(initial);initialization_seconds=time.perf_counter()-start
            initial_hash=hashlib.sha256(state_bytes(initial)).hexdigest()
            start=time.perf_counter();batches=[]
            for i in range(cfg['codec_steps']):
                ids=np.random.default_rng(np.random.SeedSequence([seed,i,30])).choice(candidates,cfg['batch_size'])
                batches.append((as_batch(tokens[ids]),jnp.asarray(canonical[ids]),jnp.asarray(importance[ids])))
            distinct=np.unique(draws['train']);replay_batch=as_batch(tokens[distinct])
            replay_ids=[jnp.asarray(np.random.default_rng(np.random.SeedSequence([seed,i,32])).integers(
                0,len(distinct),cfg['batch_size'])) for i in range(cfg['decoder_repair_steps'])]
            jax.block_until_ready((batches,replay_ids));preparation_seconds=time.perf_counter()-start
            for ai,arm in enumerate(ALL_ARMS):
                names=list(TREATMENTS);shift=(di+seed+ai)%len(names);names=names[shift:]+names[:shift]
                for treatment in names:
                    dest=directory/f'seed-{seed}'/arm/treatment;dest.mkdir(parents=True,exist_ok=resume)
                    if resume and (dest/'result.json').exists():
                        previous=json.loads((dest/'result.json').read_text())
                        assert previous['initial_state_sha256']==initial_hash
                        rows.append(previous)
                        continue
                    weights=arm_weights(arm,cfg);batch,features,iw=batches[0]
                    start=time.perf_counter();exe=compiled[length][treatment].lower(initial,batch,features,weights,iw).compile()
                    compile_seconds=time.perf_counter()-start
                    jax.block_until_ready(exe(initial,batch,features,weights,iw))
                    if resume and (dest/'codec.msgpack').exists():
                        state=serialization.from_bytes(initial,(dest/'codec.msgpack').read_bytes())
                        assert int(state.step)==cfg['codec_steps']
                        codec_seconds=None  # The interrupted run did not persist its timing.
                    else:
                        state=initial;start=time.perf_counter()
                        for batch,features,iw in batches:
                            state,metrics=exe(state,batch,features,weights,iw)
                        jax.block_until_ready((state,metrics));codec_seconds=time.perf_counter()-start
                        save_state(dest/'codec.msgpack',state)
                    before=exact_metrics(tokenizer,FrozenCodec(state.params,state.codebooks),tokens,p)
                    enc=tokenizer.apply({'params':state.params},replay_batch.tokens,replay_batch.mask,
                                        replay_batch.fallback_bytes,replay_batch.fallback_mask,state.codebooks)
                    repair=PriorState(state.params['decoder'],repair_optimizer.init(state.params['decoder']),jnp.asarray(0,jnp.int32))
                    ids=replay_ids[0];start=time.perf_counter()
                    rex=repair_fn.lower(repair,enc.representation[ids],replay_batch.tokens[ids]).compile()
                    replay_compile_seconds=time.perf_counter()-start
                    jax.block_until_ready(rex(repair,enc.representation[ids],replay_batch.tokens[ids]))
                    start=time.perf_counter()
                    for ids in replay_ids:
                        repair,loss=rex(repair,enc.representation[ids],replay_batch.tokens[ids])
                    jax.block_until_ready((repair,loss));replay_seconds=time.perf_counter()-start
                    frozen=FrozenCodec(dict(state.params,decoder=repair.params),state.codebooks)
                    after=exact_metrics(tokenizer,frozen,tokens,p);save_state(dest/'codec_repaired.msgpack',frozen)
                    # Recovery/fixed endpoints must not change encoding during decoder replay.
                    assert before['population_unique_encodings']==after['population_unique_encodings']
                    assert abs(before['factorized_oracle_kl_nats']-after['factorized_oracle_kl_nats'])<1e-9
                    tracking=state.opt_state['coarse']
                    failed=[]
                    if after['sequence_reconstruction']<.98:failed.append('reconstruction')
                    if after['factorized_missing_teacher_mass']>.01:failed.append('missing_mass')
                    if after['factorized_outside_mass']>.01:failed.append('outside_mass')
                    if after['max_sequence_reconstruction_from_codes']<.98:failed.append('collision_ceiling')
                    row=dict(domain=name,length=length,seed=seed,arm=arm,treatment=treatment,
                        before=before,after=after,failures=failed,initial_state_sha256=initial_hash,
                        codec_seconds=codec_seconds,codec_compile_seconds=compile_seconds,
                        initialization_seconds_shared=initialization_seconds,data_preparation_seconds_shared=preparation_seconds,
                        replay_seconds=replay_seconds,replay_compile_seconds=replay_compile_seconds,
                        codec_step_flops=exe.cost_analysis().get('flops'),replay_step_flops=rex.cost_analysis().get('flops'),
                        replacements=int(tracking.replacements.sum()),
                        collapsed_fraction=float(tracking.collapsed_steps)/max(int(tracking.monitored_steps),1))
                    (dest/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
                    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(json.dumps(dict(domain=name,seed=seed,arm=arm,treatment=treatment,
                        sequence=after['sequence_reconstruction'],tv=after['factorized_oracle_tv'],
                        missing=after['factorized_missing_teacher_mass'],failed=failed)),flush=True)
        print('DOMAIN_COMPLETED '+name,flush=True)
    (output/'teachers.json').write_text(json.dumps(teacher_infos,indent=2)+'\n')
    print('COMPLETED '+str(len(rows)),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    run(resume=parser.parse_args().resume)
