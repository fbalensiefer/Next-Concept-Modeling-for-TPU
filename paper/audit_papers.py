"""Check synchronized evidence, citations, references, and optional final TeX logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from generate_bibliographies import expand

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
STEMS = ('ncm_systems_paper', 'SCALE_research_paper')


def check_logs(folder):
    for stem in STEMS:
        log = (folder / (stem + '.log')).read_text(errors='replace')
        problems = re.findall(r'^.*(?:undefined|Overfull|Missing character|LaTeX Error|Emergency stop|Fatal error|Rerun to get|Label\(s\) may have changed).*$', log, re.M | re.I)
        assert not problems, (stem, problems)
        assert (folder / (stem + '.pdf')).stat().st_size > 10000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    for script in ('generate_results.py', 'generate_recent_results.py', 'generate_bibliographies.py'):
        subprocess.run([sys.executable, str(BASE / script), '--check'], check=True)
    registry = json.loads((BASE / 'references.json').read_text())
    citations = {}
    for stem in STEMS:
        source = (BASE / (stem + '.tex')).read_text()
        for common in ('shared_preamble.tex', 'shared_notation.tex', 'shared_experiments.tex'):
            assert '\\input{' + common + '}' in source, (stem, common)
        text = expand(BASE / (stem + '.tex'))
        labels = re.findall(r'\\label\{([^}]+)\}', text)
        assert len(labels) == len(set(labels)), (stem, 'duplicate labels')
        refs = set(re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}', text))
        assert refs <= set(labels), (stem, refs - set(labels))
        keys = {k.strip() for group in re.findall(r'\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]+)\}', text) for k in group.split(',')}
        assert keys <= registry.keys(), (stem, keys - registry.keys())
        citations[stem] = keys
        assert not re.search(r'eleven\s+passing|11\s+passed|no dead-code reinitialization\s+policy is implemented', text, re.I)
        # Check the shared result rather than requiring abstract wording.
        normalized=' '.join(text.split())
        assert 'reliable teacher-alignment gate was distinct' in normalized
        assert 'so that gate still failed' in normalized
        assert 'no TPU hardware was available' in text
        assert 'synthetic' in text.lower() and 'not an external pretrained' in text
        assert '0.25 times the neural Adam learning rate' in text or 'not to the neural Adam learning rate' in text
        print(f'{stem}: shared inputs, {len(labels)} unique labels, all references/citations resolved.')
    assert 'scalecompanion' in citations[STEMS[0]]
    assert 'systemscompanion' in citations[STEMS[1]]
    for line in (BASE / 'source_manifest.sha256').read_text().splitlines():
        expected, path = line.split('  ', 1)
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    validation = json.loads((BASE / 'validation.json').read_text())
    assert validation['passed'] == 44 and validation['failed'] == 0
    current=json.loads((BASE/'validation_current.json').read_text())
    assert current['passed']==62 and current['failed']==0
    if args.build_dir:
        check_logs(args.build_dir)
        print('Both final compilation logs: no undefined citations/references, missing glyphs or overfull boxes.')
    print('Cross-paper consistency checks passed.')


if __name__ == '__main__':
    main()
