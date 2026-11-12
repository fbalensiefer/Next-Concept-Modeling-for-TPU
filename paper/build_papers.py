"""Build both manuscripts, reject unresolved references, and publish final PDFs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from audit_papers import check_logs, STEMS

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('auto', 'tectonic', 'pdflatex'), default='auto')
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--offline', action='store_true', help='Tectonic: use cached resources only')
    args = parser.parse_args()
    engine = args.engine
    if engine == 'auto':
        engine = 'tectonic' if shutil.which('tectonic') else 'pdflatex'
    if not shutil.which(engine):
        raise SystemExit(f'Install {engine} (with the standard LaTeX packages listed in shared_preamble.tex).')
    if args.offline and engine != 'tectonic':
        parser.error('--offline applies only to Tectonic')
    subprocess.run([sys.executable, str(BASE / 'generate_results.py')], check=True)
    subprocess.run([sys.executable, str(BASE / 'generate_bibliographies.py')], check=True)
    with tempfile.TemporaryDirectory(prefix='ncm-papers-') as temporary:
        folder = args.build_dir.resolve() if args.build_dir else Path(temporary)
        folder.mkdir(parents=True, exist_ok=True)
        for stem in STEMS:
            if engine == 'tectonic':
                command = [engine, '--keep-logs', '--keep-intermediates', '--outdir', str(folder)]
                if args.offline:
                    command.append('--only-cached')
                subprocess.run(command + [stem + '.tex'], cwd=BASE, check=True)
            else:
                for _ in range(3):
                    subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(folder), stem + '.tex'], cwd=BASE, check=True)
        subprocess.run([sys.executable, str(BASE / 'audit_papers.py'), '--build-dir', str(folder)], check=True)
        destination = BASE.parent / 'output/pdf'
        destination.mkdir(parents=True, exist_ok=True)
        for stem in STEMS:
            shutil.copy2(folder / (stem + '.pdf'), destination / (stem + '.pdf'))
        print('Built and checked both papers in output/pdf/. Render and inspect before committing.')


if __name__ == '__main__':
    main()
