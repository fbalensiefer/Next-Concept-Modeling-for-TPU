"""Refresh provenance manifests after building and visually reviewing the PDFs."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write(name, paths):
    rows = []
    for path in sorted(set(paths)):
        rows.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(ROOT)))
    (ROOT / 'paper' / name).write_text('\n'.join(rows) + '\n')


def main():
    implementation = [path for folder in ('src/ncm', 'tests', 'examples')
                      for path in (ROOT / folder).rglob('*')
                      if path.is_file() and path.suffix in ('.py', '.json', '.txt') and '__pycache__' not in path.parts]
    write('source_manifest.sha256', implementation + [ROOT / 'pyproject.toml'])
    documents = [path for folder in ('paper', 'reports/text-experiments')
                 for path in (ROOT / folder).rglob('*')
                 if path.is_file() and path.suffix in ('.tex', '.json', '.jsonl', '.md', '.py', '.sha256')
                 and path.name != 'document_manifest.sha256' and '__pycache__' not in path.parts]
    documents += list((ROOT / 'output/pdf').glob('*.pdf'))
    documents += [ROOT / name for name in ('README.md', 'MANIFEST.in')]
    for folder in ('milestone3-benchmark', 'milestone4-scale-verified', 'scale-coverage-followup', 'scale-balanced-coverage', 'coarse-codebook-stability'):
        documents += list((ROOT / 'reports' / folder).glob('*.json'))
    write('document_manifest.sha256', documents)
    print('Implementation and document manifests refreshed; verify with shasum -a 256 -c.')


if __name__ == '__main__':
    main()
