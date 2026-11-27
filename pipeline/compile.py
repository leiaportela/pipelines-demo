import argparse
import sys
from pathlib import Path

from kfp import compiler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.definition import retail_pipeline  # noqa: E402

DEFAULT_OUT = Path(__file__).parent / "compiled" / "pipeline.json"


def compile_pipeline(out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(retail_pipeline, str(out))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(compile_pipeline(parser.parse_args().out))
