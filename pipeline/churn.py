"""Churn branch: AutoML Tabular behind train_automl, an AUC-PR gate, and batch prediction with the default model."""

from typing import NamedTuple

from google_cloud_pipeline_components.v1.automl.training_job import AutoMLTabularTrainingJobRunOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.dataset import TabularDatasetCreateOp
from google_cloud_pipeline_components.v1.model import ModelGetOp
from kfp import dsl

from pipeline import sql_builders as sb

LABELS = {"app": "vertex-demo"}
BQ_LOCATION = "US"
MODEL_DISPLAY_NAME = "vertex-demo-churn"
BUDGET_MILLI_NODE_HOURS = 1000
OPTIMIZATION_OBJECTIVE = "maximize-au-prc"
TARGET_COLUMN = "outcome"
EXPLAIN_TOP_N = 200
COLUMN_SPECS = {
    **{c: "categorical" for c in sb.CHURN_CATEGORICAL},
    **{c: "numeric" for c in sb.CHURN_NUMERIC},
}
AIPLATFORM = "google-cloud-aiplatform==1.165.1"
BASE_IMAGE = "python:3.12-slim"


@dsl.component(base_image=BASE_IMAGE)
def churn_inputs(project: str, dataset: str) -> NamedTuple(
    "Names", train_table=str, score_table=str, train_uri=str, score_uri=str, output_dataset_uri=str
):
    from collections import namedtuple

    out = namedtuple("Names", ["train_table", "score_table", "train_uri", "score_uri", "output_dataset_uri"])
    base = f"{project}.{dataset}"
    return out(
        f"`{base}.churn_automl_train`",
        f"`{base}.churn_automl_score`",
        f"bq://{base}.churn_automl_train",
        f"bq://{base}.churn_automl_score",
        f"bq://{project}.{dataset}",
    )


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-bigquery==3.47.0"])
def churn_data_check(
    project: str,
    train_query: str,
    score_query: str,
    max_invalid_fraction: float,
) -> str:
    """Fail the run before AutoML sees data it would reject: NULL, NaN or Inf feature values, or a NULL target."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    config = bigquery.QueryJobConfig(labels={"app": "vertex-demo"})
    problems = []
    for name, query in (("churn_automl_train", train_query), ("churn_automl_score", score_query)):
        row = dict(next(iter(client.query(query, job_config=config).result())).items())
        total = row["n_rows"]
        if not total:
            problems.append(f"{name}: no rows")
            continue
        for column, value in row.items():
            if column.startswith("invalid__") and value / total > max_invalid_fraction:
                problems.append(
                    f"{name}.{column[len('invalid__') :]}: {value} of {total} rows are NULL, NaN or Inf "
                    f"({value / total:.2%}, limit {max_invalid_fraction:.1%})"
                )
        if row.get("null_target"):
            problems.append(f"{name}: {row['null_target']} rows have a NULL target")
    if problems:
        raise RuntimeError("churn data check failed, no AutoML training started: " + "; ".join(problems))
    return "ok"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-bigquery==3.47.0"])
def churn_explain_input(
    project: str,
    score_table: str,
    dataset: str,
    predictions: dsl.Input[dsl.Artifact],
    target_column: str,
    at_risk_class: str,
    top_n: int,
) -> NamedTuple("Explain", table=str, uri=str, rows=int):
    """Pass 2 input: the top N customers by score of the at-risk class from pass 1, with their scoring features."""
    from collections import namedtuple

    from google.cloud import bigquery

    meta = predictions.metadata
    scored = f"`{meta['projectId']}.{meta['datasetId']}.{meta['tableId']}`"
    base = f"{project}.{dataset}"
    column = f"predicted_{target_column}"
    client = bigquery.Client(project=project)
    config = bigquery.QueryJobConfig(labels={"app": "vertex-demo"})
    client.query(
        f"""CREATE OR REPLACE TABLE `{base}.churn_explain_input` OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH ranked AS (
  SELECT SAFE_CAST(user_id AS INT64) AS user_id,
    (SELECT s FROM UNNEST({column}.scores) AS s WITH OFFSET i
     WHERE {column}.classes[OFFSET(i)] = '{at_risk_class}') AS risk
  FROM {scored}
)
SELECT t.* FROM {score_table} AS t
JOIN (SELECT user_id FROM ranked ORDER BY risk DESC, user_id LIMIT {int(top_n)}) USING (user_id)""",
        job_config=config,
    ).result()
    rows = next(
        iter(client.query(f"SELECT COUNT(*) AS n FROM `{base}.churn_explain_input`", job_config=config).result())
    )
    if rows["n"] != top_n:
        raise RuntimeError(f"expected {top_n} customers for the explained pass, found {rows['n']}")
    out = namedtuple("Explain", ["table", "uri", "rows"])
    return out(f"`{base}.churn_explain_input`", f"bq://{base}.churn_explain_input", int(rows["n"]))


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[AIPLATFORM])
def resolve_default_model(project: str, location: str, model_display_name: str) -> NamedTuple(
    "Default", model_name=str, default_version=str
):
    """Find the stable churn model in Model Registry; empty strings when it doesn't exist yet."""
    from collections import namedtuple

    from google.cloud import aiplatform

    out = namedtuple("Default", ["model_name", "default_version"])
    aiplatform.init(project=project, location=location)
    found = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    if not found:
        return out("", "")
    model = found[0]
    return out(model.resource_name, model.version_id)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[AIPLATFORM])
def churn_gate(
    project: str,
    location: str,
    model_display_name: str,
    positive_class: str,
    evaluation: dsl.Output[dsl.Artifact],
    trained: bool,
) -> NamedTuple(
    "Gate",
    model_name=str,
    version_id=str,
    versioned_name=str,
    promoted=bool,
    auprc=float,
    previous_auprc=float,
):
    """Train path (`trained`): compare the new version's AUC-PR with the default's and promote it only if it is better.

    Reuse path: require an existing default and report its evaluation. Either way the
    evaluation of the version that ends up as default is written to `evaluation` as JSON.
    """
    import json
    import time
    from collections import namedtuple

    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1 import ModelServiceClient
    from google.protobuf import json_format

    out = namedtuple("Gate", ["model_name", "version_id", "versioned_name", "promoted", "auprc", "previous_auprc"])
    aiplatform.init(project=project, location=location)
    client = ModelServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})

    def metrics_of(versioned_name):
        """AUC-PR and the curves for the positive class, from the evaluation AutoML wrote with the model."""
        evals = []
        for _ in range(6):
            evals = list(client.list_model_evaluations(parent=versioned_name))
            if evals:
                break
            time.sleep(20)
        if not evals:
            raise RuntimeError(f"no model evaluation found for {versioned_name}")
        ev = evals[0]
        overall = json_format.MessageToDict(ev._pb.metrics)
        chosen, slice_found = overall, False
        try:
            for sl in client.list_model_evaluation_slices(parent=ev.name):
                spec = sl.slice_
                if spec.dimension == "annotationSpec" and spec.value == positive_class:
                    chosen, slice_found = json_format.MessageToDict(sl._pb.metrics), True
                    break
        except Exception as exc:
            print(f"could not read evaluation slices: {exc}")
        return ev, overall, chosen, slice_found

    def number(d, key):
        v = d.get(key)
        return float(v) if v is not None else None

    def describe(versioned_name, model_name, version_id):
        ev, overall, chosen, slice_found = metrics_of(versioned_name)
        points = chosen.get("confidenceMetrics", [])
        info = {
            "model_name": model_name,
            "version_id": version_id,
            "positive_class": positive_class,
            "positive_class_slice_found": slice_found,
            "auprc": number(chosen, "auPrc"),
            "auroc": number(chosen, "auRoc"),
            "log_loss": number(chosen, "logLoss"),
            "overall_auprc": number(overall, "auPrc"),
            "pr_curve": [
                {"threshold": p.get("confidenceThreshold"), "recall": p.get("recall"), "precision": p.get("precision")}
                for p in points
            ],
            "roc_curve": [
                {
                    "threshold": p.get("confidenceThreshold"),
                    "fpr": p.get("falsePositiveRate"),
                    "tpr": p.get("recall"),
                }
                for p in points
            ],
            "confusion": [
                {
                    "threshold": p.get("confidenceThreshold"),
                    "tp": p.get("truePositiveCount"),
                    "fp": p.get("falsePositiveCount"),
                    "tn": p.get("trueNegativeCount"),
                    "fn": p.get("falseNegativeCount"),
                }
                for p in points
            ],
        }
        if points:
            best = max(points, key=lambda p: p.get("f1Score") or 0)
            info["operating_point"] = {
                "threshold": best.get("confidenceThreshold"),
                "precision": best.get("precision"),
                "recall": best.get("recall"),
                "f1": best.get("f1Score"),
            }
        try:
            expl = json_format.MessageToDict(ev._pb.model_explanation)
            attrs = expl["meanAttributions"][0]["featureAttributions"]
            ranked = sorted(attrs.items(), key=lambda kv: -abs(float(kv[1])))
            info["feature_importance"] = [
                {"feature": k, "importance": float(v), "rank": i + 1} for i, (k, v) in enumerate(ranked)
            ]
        except Exception as exc:
            info["feature_importance_error"] = str(exc)
        try:
            model = client.get_model(name=versioned_name)
            if model.training_pipeline:
                from google.cloud.aiplatform_v1 import PipelineServiceClient

                pipe = PipelineServiceClient(
                    client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
                ).get_training_pipeline(name=model.training_pipeline)
                inputs = json_format.MessageToDict(pipe._pb.training_task_inputs)
                info["training"] = {
                    "pipeline": pipe.name,
                    "duration_seconds": (pipe.end_time - pipe.start_time).total_seconds(),
                    "budget_milli_node_hours": inputs.get("budgetMilliNodeHours"),
                    "objective": inputs.get("optimizationObjective"),
                    "trained_at": pipe.end_time.isoformat(),
                }
        except Exception as exc:
            info["training_error"] = str(exc)
        return info

    found = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
    default = found[0] if found else None

    if not trained:
        if default is None:
            raise RuntimeError(
                f"No default churn model named '{model_display_name}' exists in Model Registry, "
                "so train_automl=false cannot batch-predict. Run once with train_automl=true first."
            )
        model_name, version_id, promoted = default.resource_name, default.version_id, False
        info = describe(f"{model_name}@{version_id}", model_name, version_id)
        auprc = previous = info["auprc"]
    else:
        if default is None:
            raise RuntimeError(f"training finished but no model named '{model_display_name}' exists in Model Registry")
        model_name = default.resource_name
        versions = aiplatform.ModelRegistry(model_name).list_versions()
        version_id = max(versions, key=lambda v: int(v.version_id)).version_id
        print(f"trained {model_name} version {version_id}")
        info = describe(f"{model_name}@{version_id}", model_name, version_id)
        auprc = info["auprc"]
        previous = None
        if default is None or (default.resource_name == model_name and default.version_id == version_id):
            promoted = True
        else:
            previous = describe(f"{default.resource_name}@{default.version_id}", model_name, default.version_id)[
                "auprc"
            ]
            if auprc is None or previous is None:
                raise RuntimeError(f"cannot compare AUC-PR: new={auprc}, default={previous}")
            promoted = auprc > previous
            if promoted:
                aiplatform.ModelRegistry(model_name).add_version_aliases(["default"], version_id)
            else:
                model_name, version_id = default.resource_name, default.version_id
                info = describe(f"{model_name}@{version_id}", model_name, version_id)
                auprc = info["auprc"]
    info["gate"] = {"promoted": promoted, "previous_default_auprc": previous}
    with open(evaluation.path, "w") as f:
        json.dump(info, f)
    short_ref = f"{model_name.split('/')[-1]}@{version_id}"
    return out(model_name, version_id, short_ref, promoted, float(auprc or 0.0), float(previous or 0.0))


def churn_branch_ops(project, location, bq_dataset, train_automl, churn_train, churn_score, run_date):
    """Wire the churn branch; returns the predictions table and the gate tasks for the caller."""
    names = churn_inputs(project=project, dataset=bq_dataset).set_caching_options(False)
    builds = [
        BigqueryQueryJobOp(
            query=_sql(builder(), names, churn_train, churn_score),
            project=project,
            location=BQ_LOCATION,
            labels=LABELS,
        )
        .set_display_name(f"build_{name}")
        .set_caching_options(False)
        for name, builder in sb.CHURN_BUILDERS.items()
    ]
    current = (
        resolve_default_model(project=project, location=location, model_display_name=MODEL_DISPLAY_NAME)
        .set_display_name("resolve_default_model")
        .set_caching_options(False)
    )
    check = (
        churn_data_check(
            project=project,
            train_query=_sql(sb.automl_train_check(), names, churn_train, churn_score),
            score_query=_sql(sb.automl_score_check(), names, churn_train, churn_score),
            max_invalid_fraction=sb.MAX_INVALID_FRACTION,
        )
        .set_display_name("check_automl_data")
        .set_caching_options(False)
    )
    for b in builds:
        check.after(b)
    current.after(check)
    return names, check, current


def _sql(template, names, churn_train, churn_score):
    import re

    values = {
        "churn_automl_train": names.outputs["train_table"],
        "churn_automl_score": names.outputs["score_table"],
        "churn_train": churn_train,
        "churn_score": churn_score,
    }
    pieces = [p for p in re.split(r"(\{\w+\})", template) if p]
    return "".join(f"{values[p[1:-1]]}" if p[1:-1] in values else p for p in pieces)


@dsl.pipeline(name="churn-automl")
def churn_branch(
    project: str,
    location: str,
    bq_dataset: str,
    train_automl: bool,
    churn_train: str,
    churn_score: str,
) -> NamedTuple(
    "Churn",
    model_name=str,
    version_id=str,
    evaluation=dsl.Artifact,
    predictions=dsl.Artifact,
    explanations=dsl.Artifact,
):
    names, check, current = churn_branch_ops(project, location, bq_dataset, train_automl, churn_train, churn_score, "")
    with dsl.If(train_automl == True, name="train-automl"):  # noqa: E712
        dataset = TabularDatasetCreateOp(
            project=project,
            location=location,
            display_name=MODEL_DISPLAY_NAME,
            bq_source=names.outputs["train_uri"],
            labels=LABELS,
        ).set_caching_options(False)
        dataset.after(check)
        training = AutoMLTabularTrainingJobRunOp(
            project=project,
            location=location,
            display_name=f"{MODEL_DISPLAY_NAME}-automl",
            optimization_prediction_type="classification",
            optimization_objective=OPTIMIZATION_OBJECTIVE,
            dataset=dataset.outputs["dataset"],
            target_column=TARGET_COLUMN,
            column_specs=COLUMN_SPECS,
            budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
            model_display_name=MODEL_DISPLAY_NAME,
            parent_model=current.outputs["model_name"],
            is_default_version=False,
            labels=LABELS,
            model_labels=LABELS,
            disable_early_stopping=False,
        ).set_caching_options(False)
        trained_gate = (
            churn_gate(
                project=project,
                location=location,
                model_display_name=MODEL_DISPLAY_NAME,
                positive_class=sb.POSITIVE_CLASS,
                trained=True,
            )
            .set_display_name("gate_new_version")
            .set_caching_options(False)
            .after(training)
        )
    with dsl.Else(name="reuse-default"):
        reuse_gate = (
            churn_gate(
                project=project,
                location=location,
                model_display_name=MODEL_DISPLAY_NAME,
                positive_class=sb.POSITIVE_CLASS,
                trained=False,
            )
            .set_display_name("require_default_model")
            .set_caching_options(False)
        )
        reuse_gate.after(check, current)
    model_name = dsl.OneOf(trained_gate.outputs["model_name"], reuse_gate.outputs["model_name"])
    version_id = dsl.OneOf(trained_gate.outputs["version_id"], reuse_gate.outputs["version_id"])
    evaluation = dsl.OneOf(trained_gate.outputs["evaluation"], reuse_gate.outputs["evaluation"])
    versioned_name = dsl.OneOf(trained_gate.outputs["versioned_name"], reuse_gate.outputs["versioned_name"])
    model = (
        ModelGetOp(model_name=versioned_name, project=project, location=location)
        .set_display_name("get_default_model")
        .set_caching_options(False)
    )
    model.after(check)
    batch = ModelBatchPredictOp(
        project=project,
        location=location,
        job_display_name=f"{MODEL_DISPLAY_NAME}-batch-predict",
        model=model.outputs["model"],
        instances_format="bigquery",
        predictions_format="bigquery",
        bigquery_source_input_uri=names.outputs["score_uri"],
        bigquery_destination_output_uri=names.outputs["output_dataset_uri"],
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=6,
        generate_explanation=False,
        labels=LABELS,
    ).set_caching_options(False)
    batch.set_display_name("batch_predict_all")
    top = (
        churn_explain_input(
            project=project,
            score_table=names.outputs["score_table"],
            dataset=bq_dataset,
            predictions=batch.outputs["bigquery_output_table"],
            target_column=TARGET_COLUMN,
            at_risk_class=sb.NEGATIVE_CLASS,
            top_n=EXPLAIN_TOP_N,
        )
        .set_display_name("select_top_customers")
        .set_caching_options(False)
    )
    explained = (
        ModelBatchPredictOp(
            project=project,
            location=location,
            job_display_name=f"{MODEL_DISPLAY_NAME}-batch-explain",
            model=model.outputs["model"],
            instances_format="bigquery",
            predictions_format="bigquery",
            bigquery_source_input_uri=top.outputs["uri"],
            bigquery_destination_output_uri=names.outputs["output_dataset_uri"],
            machine_type="n1-standard-4",
            starting_replica_count=1,
            max_replica_count=2,
            generate_explanation=True,
            labels=LABELS,
        )
        .set_display_name("batch_predict_explained")
        .set_caching_options(False)
    )
    return NamedTuple(
        "Churn",
        model_name=str,
        version_id=str,
        evaluation=dsl.Artifact,
        predictions=dsl.Artifact,
        explanations=dsl.Artifact,
    )(
        model_name,
        version_id,
        evaluation,
        batch.outputs["bigquery_output_table"],
        explained.outputs["bigquery_output_table"],
    )
