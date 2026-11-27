"""Build the CLV trainer source distribution and copy it to the project's Cloud Storage bucket."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.clv import TRAINER_PACKAGE

TRAINER_DIR = Path(__file__).resolve().parent.parent / "trainer"


def build(out_dir: Path) -> Path:
    """Build from a copy so the egg-info directory never lands in the repository."""
    source = out_dir / "src"
    shutil.copytree(TRAINER_DIR, source, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build"))
    subprocess.run(
        [sys.executable, "setup.py", "-q", "sdist", "--dist-dir", str(out_dir)],
        cwd=source,
        check=True,
    )
    path = out_dir / TRAINER_PACKAGE
    if not path.exists():
        raise FileNotFoundError(f"{path} was not built; found {sorted(p.name for p in out_dir.iterdir())}")
    return path


def stage(project: str) -> str:
    from google.cloud import storage

    bucket = f"{project}-vertex-demo"
    with tempfile.TemporaryDirectory() as tmp:
        package = build(Path(tmp))
        blob = storage.Client(project=project).bucket(bucket).blob(f"trainer/{TRAINER_PACKAGE}")
        blob.upload_from_filename(str(package))
    return f"gs://{bucket}/trainer/{TRAINER_PACKAGE}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("PROJECT"), help="default: $PROJECT (deploy/env.sh)")
    project = parser.parse_args().project
    if not project:
        parser.error("no project: pass --project or run `source deploy/env.sh` first")
    print(stage(project))


if __name__ == "__main__":
    main()
