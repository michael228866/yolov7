"""Bundle the yolov7 conda env into a self-contained `release/` folder.

After running this, zip the `release/` folder and ship it to a colleague.
They just unzip and double-click `start.bat` — no Python/conda needed on
their side (only a Windows machine with an NVIDIA GPU driver installed).

Usage:
    python make_release.py
    python make_release.py --out my_release    # custom output folder
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ENV_NAME = 'yolov7'

# What to ship — minimal, only what count_inout_head.py needs at runtime.
FILES_TO_COPY = [
    'count_inout_head.py',
    'db.py',
    'query_daily.py',
    'yolov8_head_medium.pt',
    'yolov8_head_nano.pt',
]


START_BAT = r"""@echo off
REM Activate bundled env and launch the head counter
cd /d "%~dp0"
call env\Scripts\activate.bat
python count_inout_head.py %*
pause
"""

QUERY_BAT = r"""@echo off
REM Print today's IN/OUT report
cd /d "%~dp0"
call env\Scripts\activate.bat
python query_daily.py %*
pause
"""

README = """\
======================================================================
 Head Counter - Quick Start  (Windows + NVIDIA GPU)
======================================================================

REQUIREMENTS
  - Windows 10/11 64-bit
  - NVIDIA GPU + driver (any recent version supports CUDA 11.8)
  - ~10 GB free disk space

WHAT'S IN THIS FOLDER
  env\                       Bundled Python environment (do NOT touch)
  count_inout_head.py        Main program
  db.py                      Database helper
  query_daily.py             Report tool
  yolov8_head_medium.pt      Head detection model (accurate, default)
  yolov8_head_nano.pt        Head detection model (faster, less accurate)
  start.bat                  Double-click to launch counter
  query.bat                  Double-click to view today's report
  counter.db                 Will be auto-created after first run

HOW TO USE
  1. Double-click `start.bat`. A camera window opens.
  2. Click 2 points on the first frame to draw the counting line.
  3. Press any key to start counting.
  4. Watch IN / OUT / TOTAL update top-left as people cross the line.
  5. Press 'q' or Esc to quit. Stats also saved into counter.db.

TO SEE TODAY'S NUMBERS
  Double-click `query.bat`.

ADVANCED OPTIONS (edit start.bat or run from command line)
  start.bat --line v                  vertical middle line, no clicking
  start.bat --line h                  horizontal middle line
  start.bat --weights yolov8_head_nano.pt   faster model
  start.bat --source path\to\video.mp4      run on a video file
  start.bat --conf-thres 0.5          lower threshold (catches more)

BACKUP YOUR DATA
  Just copy `counter.db` somewhere safe. That's the whole database.

======================================================================
"""


def run(cmd, **kw):
    print(f'>>> {cmd}')
    subprocess.check_call(cmd, shell=True, **kw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='release', help='output folder (default: release)')
    parser.add_argument('--keep-archive', action='store_true',
                        help='keep the intermediate env.tar.gz')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = (here / args.out).resolve()
    env_dir = out_dir / 'env'
    archive = out_dir / 'env.tar.gz'

    # 0. Sanity-check source files
    missing = [f for f in FILES_TO_COPY if not (here / f).exists()]
    if missing:
        sys.exit(f"Missing required files: {missing}")

    # 1. Fresh release folder
    if out_dir.exists():
        print(f'Cleaning existing {out_dir} ...')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env_dir.mkdir()

    # 2. conda-pack the env to a tarball
    print(f'\n=== Packing conda env "{ENV_NAME}" (this takes a few minutes) ===')
    run(f'conda pack -n {ENV_NAME} -o "{archive}" --ignore-missing-files --n-threads -1')

    # 3. Extract tarball into env_dir
    print(f'\n=== Extracting env into {env_dir} ===')
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(env_dir)

    if not args.keep_archive:
        archive.unlink()

    # 4. Run conda-unpack to fix shebangs / hard paths in the extracted env
    print('\n=== Running conda-unpack (rewrites internal paths) ===')
    unpack_exe = env_dir / 'Scripts' / 'conda-unpack.exe'
    if not unpack_exe.exists():
        sys.exit(f"conda-unpack not found at {unpack_exe}")
    run(f'"{unpack_exe}"')

    # 5. Copy code + weights
    print('\n=== Copying scripts and weights ===')
    for fname in FILES_TO_COPY:
        src = here / fname
        dst = out_dir / fname
        print(f'  {fname}  ({src.stat().st_size / 1024 / 1024:.1f} MB)')
        shutil.copy2(src, dst)

    # 6. Write batch launchers + README
    print('\n=== Writing launchers and README ===')
    (out_dir / 'start.bat').write_text(START_BAT, encoding='utf-8')
    (out_dir / 'query.bat').write_text(QUERY_BAT, encoding='utf-8')
    (out_dir / 'README.txt').write_text(README, encoding='utf-8')

    # 7. Done
    size_mb = sum(p.stat().st_size for p in out_dir.rglob('*') if p.is_file()) / 1024 / 1024
    print(f'\nDone. {out_dir}  ({size_mb / 1024:.1f} GB)')
    print('Next: right-click the release folder -> Send to -> Compressed (zipped) folder')
    print('      ship the resulting .zip to your colleague.')


if __name__ == '__main__':
    main()
