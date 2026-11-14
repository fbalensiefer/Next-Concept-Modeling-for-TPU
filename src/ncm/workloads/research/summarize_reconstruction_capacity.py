"""Publish the bounded capacity experiment's immutable metrics and decision."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.run), Path(args.output)
    rows = json.loads((source/'summary.json').read_text())
    if len(rows) != 6 or any(row['records'][-1]['step'] != 2000 for row in rows):
        raise ValueError('Require all six complete prespecified fits')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('protocol.json','summary.json','controls.json'):
        shutil.copyfile(source/name, output/name)
    lines = ['# Fixed-length reconstruction: duration and code capacity', '',
        'Six fresh CPU fits use the same 128-byte windows, training orders and seeds as paired controls. '
        'Codebook size (16 versus 64) is the sole architecture change. All runs use 2,000 updates, '
        'batch size four, reconstruction-first losses and the final checkpoint. Validation remains descriptive; '
        'test reconstruction is unevaluated. Source snapshots are preserved under the run directory.', '',
        '| Codebook | Seed | Step | Train byte accuracy | Validation byte accuracy | Validation exact windows | Punctuation accuracy |',
        '|---|---|---|---|---|---|---|']
    for row in rows:
        for record in row['records']:
            if record['step'] not in (100, 500, 1000, 2000):
                continue
            train, valid = record['train'], record['validation']
            lines.append(f"| {row['codebook_size']} | {row['seed']} | {record['step']} | "
                f"{train['byte_accuracy']:.4%} | {valid['byte_accuracy']:.4%} | "
                f"{valid['exact_window_accuracy']:.4%} | {valid['slices']['punctuation']['accuracy']:.4%} |")
    lines += ['', '## Paired descriptive effects', '']
    for size in (16, 64):
        selected = [r for r in rows if r['codebook_size'] == size]
        change = [r['records'][-1]['validation']['byte_accuracy']-r['records'][0]['validation']['byte_accuracy'] for r in selected]
        lines.append(f'- {size} entries, 100 to 2,000 updates: mean validation byte accuracy change '
                     f'{100*np.mean(change):.2f} percentage points; seed range {100*min(change):.2f} to {100*max(change):.2f}.')
    paired = []
    for seed in (0,1,2):
        small, large = [next(r for r in rows if r['seed']==seed and r['codebook_size']==size) for size in (16,64)]
        paired.append(large['records'][-1]['validation']['byte_accuracy']-small['records'][-1]['validation']['byte_accuracy'])
    lines += [f'- 64 versus 16 entries at 2,000 updates: mean paired difference {100*np.mean(paired):.2f} '
              f'percentage points; seed range {100*min(paired):.2f} to {100*max(paired):.2f}.', '',
        'These ranges describe three model seeds on one fixed validation book. They are not population '
        'confidence intervals or independent corpus replications. The follow-up was chosen after observing '
        'the initial pilot, so its findings are exploratory.', '', '## Cost and controls', '',
        '| Representation | Nominal bits/input byte | Reconstruction training byte exposure |',
        '|---|---|---|', '| Raw bytes | 8 | 0 |', '| Fixed byte pairs | 8 | 0 |',
        '| RVQ, 16 entries | 7 | 1,024,000 per fit |', '| RVQ, 64 entries | 10.5 | 1,024,000 per fit |', '',
        'Both exact controls reconstruct every train/validation window perfectly. Packing byte pairs '
        'does not compress them: the output vocabulary has 65,536 categories, whose prediction heads '
        'and sampling costs must be charged in any prior comparison. RVQ nominal bits exclude '
        'codec weights, codebook tables and framing. More codebook entries change both information '
        'capacity and optimization; this experiment does not isolate an information-theoretic cause.', '',
        '| Codebook | Seed | Codec parameters | Codebook embedding elements | Full run wall seconds |',
        '|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['codebook_size']} | {row['seed']} | {row['codec_parameters']} | "
                     f"{row['codebook_embedding_elements']} | {row['elapsed_seconds']:.1f} |")
    lines += ['', 'Wall time includes compilation, full train/validation evaluation every 100 steps and '
        'serialization. It is not a matched training-cost or generation-latency measure. No TPU was used.', '',
        '## Decision', '',
        ('Longer training improves validation byte accuracy in every seed for both codebook sizes. '
         if all(r['records'][-1]['validation']['byte_accuracy'] > r['records'][0]['validation']['byte_accuracy'] for r in rows)
         else 'Training-duration effects vary across seeds; inspect all curves. ')
        + ('The larger codebook also improves final byte accuracy in every paired seed. '
           if min(paired) > 0 else 'The larger codebook does not improve final byte accuracy in every paired seed. ')
        + 'These observations support a bounded next check of the strongest codec on independent documents '
        'and consequential error patterns. Keep the current codecs experimental; no application fidelity '
        'tolerance has been established. Raw and fixed-block controls already offer exact reconstruction '
        'without codec training; a learned codec must still demonstrate generation value.', '',
        'The remaining real-text checks include rare patterns, UTF-8 validity, independent documents, '
        'a frozen application fidelity tolerance and trained prior comparisons. No claims about generation '
        'quality or acceleration follow from these reconstruction results.', '',
        '## Reproduction', '',
        '```sh', 'cargo run --bin ncm-run -- reconstruction-capacity --output runs/reconstruction-capacity-v1',
        'cargo run --bin ncm-run -- summarize-reconstruction-capacity --run runs/reconstruction-capacity-v1 --output reports/reconstruction-capacity',
        '```', '', 'Use a fresh run directory. Requires the initial pilot’s processed files and protocol. '
        '[Protocol](protocol.json), [all checkpoint metrics](summary.json), [exact controls](controls.json).', '']
    (output/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
