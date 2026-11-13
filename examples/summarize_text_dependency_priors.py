"""Compact public report and lossless sample archive for text table priors."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib
import numpy as np
from ncm.text_data import document_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();source=Path(args.run);output=Path(args.output)
    integrity=json.loads((source/'integrity.json').read_text())
    rows=json.loads((source/'results.json').read_text())
    if len(rows)!=10 or not integrity['frozen_checkpoints_unchanged']:
        raise ValueError('Require ten complete fits and unchanged codecs')
    output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    validation_path=next(Path(p) for p in protocol['data_hashes'] if Path(p).name=='validation.jsonl')
    if hashlib.sha256(validation_path.read_bytes()).hexdigest()!=protocol['data_hashes'][str(validation_path)]:
        raise ValueError('Reference corpus hash changed')
    reference,_=document_dataset(validation_path,128)
    valid_utf8=0
    for row in reference[0]:
        try:
            bytes(row.astype(np.uint8)).decode('utf-8')
            valid_utf8+=1
        except UnicodeDecodeError:
            pass
    reference_validity=valid_utf8/len(reference[0])
    (output/'reference.json').write_text(json.dumps(dict(windows=len(reference[0]),
        valid_utf8_fraction=reference_validity),indent=2)+'\n')
    archive={};examples={}
    for row in rows:
        key=row['representation']+'_'+row['prior']
        examples[key]=[]
        for sample in row['samples']:
            stats=sample['diagnostics']
            archive[key+'_sample_seed_'+str(sample['seed'])]=np.asarray(stats.pop('sample_bytes'),np.uint8)
            text=stats.pop('sample_text')
            examples[key].append(dict(seed=sample['seed'],first_four=text[:4]))
    np.savez_compressed(output/'samples.npz',**archive)
    (output/'sample_excerpts.json').write_text(json.dumps(examples,indent=2)+'\n')
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    for name in ('protocol.json','integrity.json'):
        shutil.copyfile(source/name,output/name)
    lines=['# Real-text representation and dependency diagnostics','',
        'Ten stationary table fits compare raw bytes, reversible byte pairs and three frozen 64-entry '
        'RVQ codecs. Every representation is fitted with an independent marginal and a sparse '
        'first-order Markov prior. Each RVQ level has its own prior and no cross-level conditioning. '
        'These controls are deliberately weaker than the project’s neural coarse-conditioned prior; '
        'they test local dependence rather than replace that model.', '',
        'Training and validation reuse the book-separated 128-byte windows from the reconstruction '
        'experiments. Smoothing strength is fixed at one before fitting. All three learned-codec seeds '
        'are retained. No test reconstruction or generation evaluation is performed.', '',
        '## Held-out modeling loss', '',
        'Compare loss only between priors for the same representation and codec. RVQ code loss is '
        'not a byte likelihood. Even the reversible fixed-pair loss describes a different restricted '
        'model family; lower loss alone does not establish better generation.', '',
        '| Representation | Independent NLL/code | Markov NLL/code | Reduction |',
        '|---|---|---|---|']
    names=list(dict.fromkeys(r['representation'] for r in rows))
    reductions=[]
    for name in names:
        a,b=[next(r for r in rows if r['representation']==name and r['prior']==p) for p in ('independent','markov')]
        delta=a['validation_nll_per_code']-b['validation_nll_per_code'];reductions.append(delta)
        lines.append(f"| {name} | {a['validation_nll_per_code']:.4f} | {b['validation_nll_per_code']:.4f} | {delta:.4f} |")
    lines+=['','## Decoded sample diagnostics','',
        'Each cell averages three sampling seeds, 256 windows each. Seeds quantify Monte Carlo '
        'variation, not independent data/model fits. The raw and fixed-pair fits are deterministic '
        'given this training corpus. Bigram TV is a local distributional diagnostic; coherent '
        'language, diversity and semantic quality cannot be inferred from it alone.', '',
        '| Representation | Prior | Byte-bigram TV ↓ | Distinct windows | Valid UTF-8 windows | Equal adjacent bytes | Exact training-window matches |',
        '|---|---|---|---|---|---|---|']
    fields=['byte_bigram_tv_to_heldout','distinct_fraction','valid_utf8_fraction','adjacent_equal_byte_fraction','exact_training_window_match_fraction']
    for row in rows:
        means=[np.mean([s['diagnostics'][f] for s in row['samples']]) for f in fields]
        lines.append('| '+row['representation']+' | '+row['prior']+' | '+' | '.join(f'{x:.4f}' for x in means)+' |')
    lines+=['', f'Validation reference: {reference_validity:.4%} of its {len(reference[0])} fixed-byte windows are valid UTF-8. '
        'This establishes the boundary-cut limitation independently of generation.', '', '## Local CPU cost','',
        'Generation timing includes NumPy random sampling, all code levels, representation conversion, '
        'device transfer, neural decoding when applicable and host materialization of 16 complete '
        '128-byte windows. One warmup precedes seven timed batches. RVQ decoder compilation is '
        'already incurred during reconstruction auditing, so warmup is not a compilation measure. '
        'Raw and pair decoders are exact cheap transforms. These Python table implementations '
        'do not establish accelerator latency, peak memory, or speed relative to a cached neural AR baseline.', '',
        '| Representation | Prior | Fit seconds | Median batch ms | p90 batch ms | Numeric prior payload bytes |',
        '|---|---|---|---|---|---|']
    for row in rows:
        t=np.asarray(row['generation_seconds_batch16'])*1000
        lines.append(f"| {row['representation']} | {row['prior']} | {row['fit_wall_seconds']:.3f} | {np.median(t):.3f} | {np.percentile(t,90):.3f} | {row['numeric_prior_payload_bytes']} |")
    lines+=['','Numeric payload excludes Python-object overhead, sparse dictionary counters, codec parameters, '
        'codebooks and device/compiler buffers; it is not total or peak memory. Fit time excludes '
        'codec training and includes held-out loss computation in this runner. Original codec '
        'training costs remain in the capacity study. The stationary first-order model introduces '
        'a dependency chain; this implementation samples each level sequentially rather than '
        'optimizing parallel execution across levels. No quality–latency success claim follows.', '',
        '## Decision','',
        ('Adding adjacent dependencies reduces held-out prior loss in all five representations. '
         if min(reductions)>0 else 'Adjacent dependency gains vary by representation. ')
        +'Raw and fixed-pair Markov models have substantially lower byte-bigram discrepancy than '
        'the learned-code Markov models in this study. Inspected samples remain visibly incoherent; '
        'this is an informal observation, not a blinded quality rating. RVQ therefore has no '
        'demonstrated generation advantage here. Local dependence '
        'is a useful next modeling control, but this study does not select a winning representation '
        'or establish that RVQ earns its codec training and information cost. A stronger next comparison '
        'needs coarse-conditioned neural learned-code priors, credible cached autoregressive raw-byte '
        'generation and a prespecified quality evaluation before selecting a speed operating point.', '',
        'One training book and one validation book, stationary pooling across positions, finite '
        'sampling, and lack of independent document replications limit the evidence. Exact UTF-8 '
        'window validity also depends on arbitrary byte cuts; compare against reference validity '
        'rather than interpreting every incomplete boundary as a modeling failure. No scalable '
        'total-correlation estimate is needed to observe these controlled dependency effects.', '',
        '## Reproduction and audit','',
        '```sh', '.venv312/bin/python examples/text_dependency_priors.py --output runs/text-dependency-priors-v1',
        '.venv312/bin/python examples/summarize_text_dependency_priors.py --run runs/text-dependency-priors-v1 --output reports/text-dependency-priors',
        '```','',
        'Requires the initial processed corpus and all three capacity-study 64-entry checkpoints. '
        'Use a fresh run directory. [Protocol](protocol.json) records source/data/checkpoint hashes '
        'and software versions; exact executable sources are saved under the run directory. '
        '[All metrics and timings](results.json), [sample excerpts](sample_excerpts.json), '
        '[lossless sampled bytes](samples.npz), [frozen-codec integrity](integrity.json). '
        'Code-to-byte decoding was checked against all three recorded validation reconstruction '
        'accuracies before fitting; checkpoint hashes were checked again after every fit completed.', '']
    (output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
