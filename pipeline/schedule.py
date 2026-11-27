"""Create the weekly pipeline schedule (idempotent) and print its resource name.

The schedule is left ON. A schedule whose pipeline spec, cron or parameters differ from the compiled
pipeline is deleted and created again, so rerunning this file after a pipeline change refreshes it.
It never trains AutoML: scheduled runs use the current default churn model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.run import COMPILED, LABELS, LOCATION, PIPELINE_ROOT, PROJECT, SERVICE_ACCOUNT, require_project  # noqa: E402

DISPLAY_NAME = "vertex-demo-weekly"
CRON = "0 6 * * 1"
TIME_ZONE = "America/Sao_Paulo"
MAX_CONCURRENT_RUNS = 1
PARAMETERS = {"project": PROJECT, "location": LOCATION, "run_date": "", "train_automl": False}


def schedule_spec() -> dict:
    return {
        "display_name": DISPLAY_NAME,
        "cron": f"TZ={TIME_ZONE} {CRON}",
        "max_concurrent_run_count": MAX_CONCURRENT_RUNS,
        "service_account": SERVICE_ACCOUNT,
        "labels": LABELS,
        "parameter_values": PARAMETERS,
        "pipeline_root": PIPELINE_ROOT,
    }


def without_caching(node):
    """Vertex adds cachingOptions to every task when the job is created with caching off; drop them to compare."""
    if isinstance(node, dict):
        return {k: without_caching(v) for k, v in node.items() if k != "cachingOptions"}
    return node


def is_current(existing) -> bool:
    """True when the live schedule runs the compiled pipeline with the spec's cron and parameters."""
    from google.protobuf import json_format

    spec = schedule_spec()
    job = existing.gca_resource.create_pipeline_job_request.pipeline_job
    live = json_format.MessageToDict(job._pb.pipeline_spec)
    parameters = json_format.MessageToDict(job._pb.runtime_config)["parameterValues"]
    return (
        without_caching(live) == without_caching(json.loads(Path(COMPILED).read_text()))
        and existing.gca_resource.cron == spec["cron"]
        and parameters == spec["parameter_values"]
        and existing.gca_resource.max_concurrent_run_count == spec["max_concurrent_run_count"]
    )


def ensure():
    from google.cloud import aiplatform

    require_project()
    aiplatform.init(project=PROJECT, location=LOCATION)
    spec = schedule_spec()
    existing = [
        s
        for s in aiplatform.PipelineJobSchedule.list(filter=f'display_name="{DISPLAY_NAME}"')
        if s.gca_resource.state.name in ("ACTIVE", "PAUSED")
    ]
    for old in existing:
        if is_current(old):
            return old
        old.delete()
    job = aiplatform.PipelineJob(
        display_name="vertex-demo-retail-intelligence",
        template_path=str(COMPILED),
        pipeline_root=spec["pipeline_root"],
        parameter_values=spec["parameter_values"],
        labels=spec["labels"],
        enable_caching=False,
    )
    schedule = aiplatform.PipelineJobSchedule(pipeline_job=job, display_name=spec["display_name"])
    schedule.create(
        cron=spec["cron"],
        max_concurrent_run_count=spec["max_concurrent_run_count"],
        service_account=spec["service_account"],
    )
    return schedule


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create or refresh the weekly schedule (takes no arguments).")
    parser.parse_args(argv)
    schedule = ensure()
    print(f"schedule: {schedule.resource_name}")
    print(f"cron: {schedule_spec()['cron']} state: {schedule.gca_resource.state.name}")


if __name__ == "__main__":
    main()
