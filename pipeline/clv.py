"""CLV branch: XGBoost trainer with a Vertex AI tuning job, a Model Registry version and batch prediction."""

import re
from typing import NamedTuple

from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.model import ModelGetOp
from kfp import dsl
from kfp.dsl import Artifact, Input, Output

from pipeline import sql_builders as sb

LABELS = {"app": "vertex-demo"}
BQ_LOCATION = "US"
MODEL_DISPLAY_NAME = "vertex-demo-clv"
MAX_TRIAL_COUNT = 12
PARALLEL_TRIAL_COUNT = 3
MAX_FAILED_TRIAL_COUNT = 4
MACHINE_TYPE = "n1-standard-4"
BATCH_MACHINE_TYPE = "n1-standard-4"
TRAINER_PACKAGE = "clv_trainer-0.1.0.tar.gz"
TRAIN_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1@"
    "sha256:a2fea4ad3379535f2e93126bcf06c9ef6dc5803ac42f36fa4af11e1ef8cd3b14"
)
# Batch prediction rejects a digest-pinned image ("Please use a Model with a valid image"), so this one is a tag.
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
FEATURES = list(sb.CLV_MODEL_FEATURES)
AIPLATFORM = "google-cloud-aiplatform==1.165.1"
BASE_IMAGE = "python:3.12-slim"


@dsl.component(base_image=BASE_IMAGE)
def clv_inputs(project: str, dataset: str, run_id: str) -> NamedTuple(
    "Names",
    train_table=str,
    score_table=str,
    train_ref=str,
    export_uri=str,
    export_files=str,
    predictions_prefix=str,
    predictions_table=str,
):
    from collections import namedtuple

    out = namedtuple(
        "Names",
        [
            "train_table",
            "score_table",
            "train_ref",
            "export_uri",
            "export_files",
            "predictions_prefix",
            "predictions_table",
        ],
    )
    base = f"{project}.{dataset}"
    root = f"gs://{project}-vertex-demo/clv-batch/{run_id.rsplit('/', 1)[-1]}"
    return out(
        f"`{base}.clv_model_train`",
        f"`{base}.clv_model_score`",
        f"{base}.clv_model_train",
        f"{root}/input",
        f"{root}/input/part-*.jsonl",
        f"{root}/output",
        f"{base}.clv_predictions",
    )


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-bigquery==3.47.0", "google-cloud-storage"])
def clv_load_predictions(
    project: str,
    predictions_dir: Input[Artifact],
    table: str,
    expected_rows_table: str,
    predictions: Output[Artifact],
) -> NamedTuple("Loaded", rows=int):
    """Load the batch prediction JSONL (user_id, prediction) into BigQuery, labelled."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=[
            bigquery.SchemaField("user_id", "INT64"),
            bigquery.SchemaField("prediction", "FLOAT64"),
        ],
        ignore_unknown_values=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        labels={"app": "vertex-demo"},
    )
    uri = predictions_dir.uri.rstrip("/") + "/prediction.results-*"
    client.load_table_from_uri(uri, table, job_config=config).result()
    target = client.get_table(table)
    target.labels = {"app": "vertex-demo"}
    client.update_table(target, ["labels"])
    check = next(
        iter(
            client.query(
                f"SELECT COUNT(DISTINCT user_id) AS users, COUNTIF(prediction IS NULL) AS no_pred FROM `{table}`",
                job_config=bigquery.QueryJobConfig(labels={"app": "vertex-demo"}),
            ).result()
        )
    )
    expected = next(
        iter(
            client.query(
                f"SELECT COUNT(*) AS n FROM {expected_rows_table}",
                job_config=bigquery.QueryJobConfig(labels={"app": "vertex-demo"}),
            ).result()
        )
    ).n
    rows = target.num_rows
    if rows != expected or check.users != expected or check.no_pred:
        raise RuntimeError(
            f"batch prediction loaded {rows} rows, {check.users} distinct user_id, "
            f"{check.no_pred} NULL predictions for {expected} scored customers"
        )
    predictions.uri = f"bq://{table}"
    predictions.metadata["rows"] = rows
    return (rows,)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[AIPLATFORM])
def clv_tune(
    project: str,
    location: str,
    train_ref: str,
    features: str,
    trainer_package: str,
    train_image: str,
    machine_type: str,
    max_trial_count: int,
    parallel_trial_count: int,
    max_failed_trial_count: int,
    trials: Output[Artifact],
) -> NamedTuple(
    "Tuning",
    job_name=str,
    best_trial_id=str,
    base_output_dir=str,
    duration_seconds=float,
    trial_count=int,
):
    """Run the tuning job through the SDK (the GCPC tuning op has no labels input) and record every trial."""
    import datetime
    import json
    from collections import namedtuple

    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    labels = {"app": "vertex-demo"}
    bucket = f"{project}-vertex-demo"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"gs://{bucket}/clv/tuning/{stamp}"
    aiplatform.init(project=project, location=location, staging_bucket=f"gs://{bucket}/pipeline-root")
    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": machine_type},
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": train_image,
                "package_uris": [f"gs://{bucket}/trainer/{trainer_package}"],
                "python_module": "clv_trainer.task",
                "args": [f"--project={project}", f"--train_table={train_ref}", f"--features={features}"],
            },
        }
    ]
    custom = aiplatform.CustomJob(
        display_name="vertex-demo-clv-trial",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base,
        labels=labels,
    )
    job = aiplatform.HyperparameterTuningJob(
        display_name="vertex-demo-clv-tuning",
        custom_job=custom,
        metric_spec={"rmse": "minimize"},
        parameter_spec={
            "max_depth": hpt.IntegerParameterSpec(min=2, max=8, scale="linear"),
            "learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.3, scale="log"),
            "n_estimators": hpt.IntegerParameterSpec(min=50, max=400, scale="linear"),
            "subsample": hpt.DoubleParameterSpec(min=0.5, max=1.0, scale="linear"),
            "colsample_bytree": hpt.DoubleParameterSpec(min=0.5, max=1.0, scale="linear"),
            "min_child_weight": hpt.DoubleParameterSpec(min=1.0, max=60.0, scale="log"),
        },
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        max_failed_trial_count=max_failed_trial_count,
        labels=labels,
    )
    job.run(service_account=f"vertex-demo-pipelines@{project}.iam.gserviceaccount.com", sync=True)

    rows = []
    for t in job.trials:
        rmse = None
        if t.final_measurement and t.final_measurement.metrics:
            rmse = next((m.value for m in t.final_measurement.metrics if m.metric_id == "rmse"), None)
        rows.append(
            {
                "trial_id": str(t.id),
                "state": t.state.name,
                "hyperparameters": {p.parameter_id: p.value for p in t.parameters},
                "rmse": rmse,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "end_time": t.end_time.isoformat() if t.end_time else None,
            }
        )
    done = [r for r in rows if r["state"] == "SUCCEEDED" and r["rmse"] is not None]
    if not done:
        raise RuntimeError(f"tuning job {job.resource_name} has no succeeded trial: {json.dumps(rows)}")
    best = min(done, key=lambda r: r["rmse"])
    for r in rows:
        r["is_best"] = r["trial_id"] == best["trial_id"]
    with open(trials.path, "w") as f:
        json.dump(rows, f)
    gca = job._gca_resource
    duration = (gca.end_time - gca.start_time).total_seconds()
    out = namedtuple("Tuning", ["job_name", "best_trial_id", "base_output_dir", "duration_seconds", "trial_count"])
    return out(job.resource_name, best["trial_id"], base, float(duration), len(rows))


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[AIPLATFORM])
def clv_register(
    project: str,
    location: str,
    model_display_name: str,
    serving_image: str,
    features: str,
    machine_type: str,
    job_name: str,
    best_trial_id: str,
    base_output_dir: str,
    duration_seconds: float,
    trials: Input[Artifact],
    evaluation: Output[Artifact],
) -> NamedTuple("Registered", model_name=str, version_id=str, versioned_name=str, rmse=float, mae=float, r2=float):
    """Upload the best trial's model as a new default version of the stable model and write its evaluation."""
    import json
    from collections import namedtuple

    from google.cloud import aiplatform, storage

    labels = {"app": "vertex-demo"}
    trial_dir = f"{base_output_dir}/{best_trial_id}"
    bucket_name, prefix = trial_dir[len("gs://") :].split("/", 1)
    blob = storage.Client(project=project).bucket(bucket_name).blob(f"{prefix}/eval/metrics.json")
    metrics = json.loads(blob.download_as_text())
    with open(trials.path) as f:
        all_trials = json.load(f)

    aiplatform.init(project=project, location=location)
    existing = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    description = (
        f"XGBoost 180-day spend regression. Validation RMSE {metrics['rmse']:.3f}, MAE {metrics['mae']:.3f}, "
        f"R2 {metrics['r2']:.3f} (trial {best_trial_id} of {len(all_trials)}, tuning job {job_name})."
    )
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=f"{trial_dir}/model",
        serving_container_image_uri=serving_image,
        parent_model=existing[0].resource_name if existing else None,
        is_default_version=True,
        version_description=description,
        description=description,
        labels=labels,
        sync=True,
    )
    info = {
        "model_name": model.resource_name,
        "version_id": model.version_id,
        "model_display_name": model_display_name,
        "tuning_job": job_name,
        "best_trial_id": best_trial_id,
        "machine_type": machine_type,
        "duration_seconds": duration_seconds,
        "max_trial_count": len(all_trials),
        "features": features.split(","),
        "metrics": metrics,
        "trials": all_trials,
        "validation_predictions": f"{trial_dir}/eval/validation_predictions.csv",
    }
    with open(evaluation.path, "w") as f:
        json.dump(info, f)
    short_ref = f"{model.resource_name.split('/')[-1]}@{model.version_id}"
    out = namedtuple("Registered", ["model_name", "version_id", "versioned_name", "rmse", "mae", "r2"])
    return out(
        model.resource_name,
        model.version_id,
        short_ref,
        float(metrics["rmse"]),
        float(metrics["mae"]),
        float(metrics["r2"]),
    )


def sql_input(template: str, **values):
    pieces = [p for p in re.split(r"(\{\w+\})", template) if p]
    return "".join(f"{values[p[1:-1]]}" if p[1:-1] in values else p for p in pieces)


def _sql(template: str, names, clv_train, clv_score):
    return sql_input(
        template,
        clv_export_uri=names.outputs["export_uri"],
        clv_train=clv_train,
        clv_score=clv_score,
        clv_model_train=names.outputs["train_table"],
        clv_model_score=names.outputs["score_table"],
    )


def clv_predict_ops(project, location, names, model, builds, clv_train, clv_score):
    """Export the scoring table, batch-predict with the model, load the predictions; returns the load task."""
    export = (
        BigqueryQueryJobOp(
            query=_sql(sb.clv_score_export(), names, clv_train, clv_score),
            project=project,
            location=BQ_LOCATION,
            labels=LABELS,
        )
        .set_display_name("export_clv_score")
        .set_caching_options(False)
    )
    export.after(*builds)
    batch = ModelBatchPredictOp(
        project=project,
        location=location,
        job_display_name=f"{MODEL_DISPLAY_NAME}-batch-predict",
        model=model.outputs["model"],
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source_uris=[names.outputs["export_files"]],
        gcs_destination_output_uri_prefix=names.outputs["predictions_prefix"],
        instance_type="array",
        key_field="user_id",
        included_fields=FEATURES,
        machine_type=BATCH_MACHINE_TYPE,
        starting_replica_count=1,
        max_replica_count=4,
        generate_explanation=False,
        labels=LABELS,
    ).set_caching_options(False)
    batch.after(export)
    loaded = (
        clv_load_predictions(
            project=project,
            predictions_dir=batch.outputs["gcs_output_directory"],
            table=names.outputs["predictions_table"],
            expected_rows_table=names.outputs["score_table"],
        )
        .set_display_name("load_clv_predictions")
        .set_caching_options(False)
    )
    return loaded


@dsl.pipeline(name="clv-xgboost")
def clv_branch(
    project: str,
    location: str,
    bq_dataset: str,
    clv_train: str,
    clv_score: str,
) -> NamedTuple("Clv", model_name=str, version_id=str, evaluation=dsl.Artifact, predictions=dsl.Artifact):
    names = clv_inputs(
        project=project, dataset=bq_dataset, run_id=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER
    ).set_caching_options(False)
    builds = [
        BigqueryQueryJobOp(
            query=_sql(builder(), names, clv_train, clv_score), project=project, location=BQ_LOCATION, labels=LABELS
        )
        .set_display_name(f"build_{name}")
        .set_caching_options(False)
        for name, builder in sb.CLV_BUILDERS.items()
    ]
    tuning = (
        clv_tune(
            project=project,
            location=location,
            train_ref=names.outputs["train_ref"],
            features=",".join(FEATURES),
            trainer_package=TRAINER_PACKAGE,
            train_image=TRAIN_IMAGE,
            machine_type=MACHINE_TYPE,
            max_trial_count=MAX_TRIAL_COUNT,
            parallel_trial_count=PARALLEL_TRIAL_COUNT,
            max_failed_trial_count=MAX_FAILED_TRIAL_COUNT,
        )
        .set_display_name("tune_xgboost")
        .set_caching_options(False)
    )
    for b in builds:
        tuning.after(b)
    registered = (
        clv_register(
            project=project,
            location=location,
            model_display_name=MODEL_DISPLAY_NAME,
            serving_image=SERVING_IMAGE,
            features=",".join(FEATURES),
            machine_type=MACHINE_TYPE,
            job_name=tuning.outputs["job_name"],
            best_trial_id=tuning.outputs["best_trial_id"],
            base_output_dir=tuning.outputs["base_output_dir"],
            duration_seconds=tuning.outputs["duration_seconds"],
            trials=tuning.outputs["trials"],
        )
        .set_display_name("register_best_trial")
        .set_caching_options(False)
    )
    model = (
        ModelGetOp(model_name=registered.outputs["versioned_name"], project=project, location=location)
        .set_display_name("get_clv_model")
        .set_caching_options(False)
    )
    loaded = clv_predict_ops(project, location, names, model, builds, clv_train, clv_score)
    return NamedTuple("Clv", model_name=str, version_id=str, evaluation=dsl.Artifact, predictions=dsl.Artifact)(
        registered.outputs["model_name"],
        registered.outputs["version_id"],
        registered.outputs["evaluation"],
        loaded.outputs["predictions"],
    )
