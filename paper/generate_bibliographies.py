"""Emit only cited entries from one shared bibliographic registry; no BibTeX needed."""
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent


def expand(path, seen=None):
    seen = set() if seen is None else seen
    if path in seen or path.name.startswith('bibliography_'):
        return ''
    seen.add(path)
    text = path.read_text()
    return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(BASE / (m[1] if m[1].endswith('.tex') else m[1]+'.tex'), seen), text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    entries = json.loads((BASE / 'references.json').read_text())
    for stem, suffix in [('ncm_systems_paper','systems'), ('SCALE_research_paper','scale')]:
        text = expand(BASE / (stem+'.tex'))
        keys = []
        for match in re.finditer(r'\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]+)\}', text):
            for key in match[1].split(','):
                key = key.strip()
                if key not in keys:
                    keys.append(key)
        missing = set(keys) - entries.keys()
        if missing:
            raise SystemExit(f'Missing bibliography entries: {missing}')
        output = '% Generated from references.json; do not hand-edit.\n\\begingroup\n\\raggedright\n\\begin{thebibliography}{99}\n' + '\n\n'.join(entries[k] for k in keys) + '\n\\end{thebibliography}\n\\endgroup\n'
        path = BASE / 'generated' / ('bibliography_'+suffix+'.tex')
        if args.check:
            assert path.read_text() == output, path
        else:
            path.write_text(output)
        print(f'{stem}: {len(keys)} citations resolved from the shared registry.')


if __name__ == '__main__':
    main()
