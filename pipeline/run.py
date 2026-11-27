import argparse
import os
import re
import sys
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Project and region come from deploy/env.sh; the service account and bucket follow deploy/setup-gcp.sh's names.
PROJECT = os.environ.get("PROJECT", "")
LOCATION = os.environ.get("REGION", "us-central1")
SERVICE_ACCOUNT = f"vertex-demo-pipelines@{PROJECT}.iam.gserviceaccount.com"
PIPELINE_ROOT = f"gs://{PROJECT}-vertex-demo/pipeline-root"

LABELS = {"app": "vertex-demo"}
COMPILED = Path(__file__).parent / "compiled" / "pipeline.json"


def require_project() -> None:
    if not PROJECT:
        sys.exit("PROJECT is not set: run `source deploy/env.sh` first")


def submit(train_automl: bool = False, run_date: str = "", job_id: str | None = None) -> aiplatform.PipelineJob:
    from pipeline.stage_trainer import stage

    require_project()
    print(f"trainer package: {stage(PROJECT)}")
    aiplatform.init(project=PROJECT, location=LOCATION)
    job = aiplatform.PipelineJob(
        display_name="vertex-demo-retail-intelligence",
        template_path=str(COMPILED),
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project": PROJECT,
            "location": LOCATION,
            "run_date": run_date,
            "train_automl": train_automl,
        },
        labels=LABELS,
        job_id=job_id,
        enable_caching=False,
    )
    job.submit(service_account=SERVICE_ACCOUNT)
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-automl", action="store_true", default=False)
    parser.add_argument("--run-date", default="")
    parser.add_argument("--job-id", default=None, help="run id: lowercase letters, digits and hyphens")
    args = parser.parse_args()
    if args.job_id is not None and not re.fullmatch(r"[a-z][a-z0-9-]{0,127}", args.job_id):
        parser.error("--job-id must start with a lowercase letter and use only lowercase letters, digits and hyphens")
    job = submit(train_automl=args.train_automl, run_date=args.run_date, job_id=args.job_id)
    print(f"run id: {job.job_id}")
    print(
        f"console: https://console.cloud.google.com/vertex-ai/locations/{LOCATION}/pipelines/runs/{job.job_id}?project={PROJECT}"
    )
