"""Provenance and paired statistics for local validation studies (host-side)."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def identity():
    root=Path(__file__).resolve().parents[2]
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root/'src/ncm').glob('*.py'))})


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(**identity(), **data),indent=2,allow_nan=False)+'\n')


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    commit=identity()['git_commit']
    rows=[dict(git_commit=commit,**row) for row in rows]
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['git_commit']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def paired_bootstrap(a,b,seed=202,repetitions=10000):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not len(a): raise ValueError('paired observations required')
    delta=a-b
    indices=np.random.default_rng(seed).integers(len(delta),size=(repetitions,len(delta)))
    low,high=np.percentile(delta[indices].mean(1),[2.5,97.5])
    return dict(mean_difference=float(delta.mean()),ci95=[float(low),float(high)],
                seeds=len(delta),label='exploratory' if len(delta)<10 else 'bootstrap')
