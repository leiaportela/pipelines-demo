"""Publish step: writes every contract table of vertex_demo_dashboard for one run.

`publish_results` builds the churn, CLV and demand tables from the branch outputs and writes them once the
branches are done. `record_run` is the exit handler: it always writes the `runs` row and the `step_timings`
rows, so failed runs show up in the history. Both validate every row against the schema (passed in as JSON from
`publish_schema.py`) and fail on a mismatch, delete the run's rows first so a retry is idempotent, and label
every BigQuery job `app=vertex-demo`. Nothing outside the dashboard dataset is written.

Component sources run standalone in the container, so each component carries its own helpers.
"""

import json

from kfp import dsl
from kfp.dsl import Artifact, Input

from pipeline import publish_schema

BASE_IMAGE = "python:3.12-slim"
PACKAGES = ["google-cloud-bigquery==3.47.0", "google-cloud-aiplatform==1.165.1", "google-cloud-storage"]
DEMAND_ACTUAL_DAYS = 120
CLV_SAMPLE_ROWS = 500
TOP_CUSTOMERS = 30


def schema_json() -> str:
    return json.dumps(publish_schema.TABLES, sort_keys=True)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def record_run(
    project: str,
    location: str,
    job_resource_name: str,
    run_id: str,
    run_date: str,
    train_automl: bool,
    dashboard_dataset: str,
    schema_json: str,
    status: dsl.PipelineTaskFinalStatus,
) -> int:
    """Exit handler: write the run row with its final state and one step_timings row per finished task."""
    if dashboard_dataset != "vertex_demo_dashboard":
        raise ValueError(f"publish writes only vertex_demo_dashboard, not {dashboard_dataset!r}")
    run_id = run_id.rsplit("/", 1)[-1]  # the job id placeholder is an internal number, the resource name carries the id
    import datetime
    import json
    import math

    from google.cloud import aiplatform, bigquery
    from google.protobuf import json_format

    schema = json.loads(schema_json)
    labels = {"app": "vertex-demo"}
    client = bigquery.Client(project=project)

    def validate(table, rows):
        types = {f["name"]: f for f in schema[table]["fields"]}
        for row in rows:
            for name in row:
                if name not in types:
                    raise ValueError(f"{table}: column {name!r} is not in the publish schema")
            for name, field in types.items():
                value = row.get(name)
                if value is None:
                    if field["mode"] == "REQUIRED":
                        raise ValueError(f"{table}: required column {name!r} is missing")
                    continue
                kind = field["type"]
                ok = {
                    "STRING": isinstance(value, str),
                    "INT64": isinstance(value, int) and not isinstance(value, bool),
                    "FLOAT64": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                    "BOOL": isinstance(value, bool),
                    "TIMESTAMP": isinstance(value, str),
                    "DATE": isinstance(value, str),
                    "JSON": isinstance(value, (dict, list, str)),
                }[kind]
                if not ok:
                    raise ValueError(f"{table}.{name}: {value!r} is not a {kind}")

    def write(table, rows):
        validate(table, rows)
        fields = [
            bigquery.SchemaField(f["name"], f["type"], mode=f["mode"], description=f["description"])
            for f in schema[table]["fields"]
        ]
        ref = f"{project}.{dashboard_dataset}.{table}"
        created = bigquery.Table(ref, schema=fields)
        created.labels = labels
        created.description = schema[table]["description"]
        client.create_table(created, exists_ok=True)
        config = bigquery.QueryJobConfig(
            labels=labels, query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        )
        client.query(f"DELETE FROM `{ref}` WHERE run_id = @run_id", job_config=config).result()
        if rows:
            json_columns = {f["name"] for f in schema[table]["fields"] if f["type"] == "JSON"}
            # the client serializes each row once; a JSON column must hold the object, or it is stored as a string of it
            body = [
                {k: (json.loads(v) if k in json_columns and isinstance(v, str) else v) for k, v in r.items()}
                for r in rows
            ]
            load = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                schema=fields,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                labels=labels,
            )
            client.load_table_from_json(body, ref, job_config=load).result()

    def iso(value):
        return value.isoformat() if value else None

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob.get(job_resource_name)
    spec = json_format.MessageToDict(job.gca_resource._pb.pipeline_spec)
    branches = {
        "build-features": "shared",
        "churn-automl": "churn",
        "clv-xgboost": "clv",
        "demand-bqml-arima": "demand",
    }
    helpers = {"churn-inputs", "clv-inputs", "resolve-names", "forecast-names", "record-run"}
    wrappers = {name for name, comp in spec["components"].items() if "dag" in comp}
    display = {}
    for comp_name, comp in list(spec["components"].items()) + [("root", spec["root"])]:
        for task_name, task in comp.get("dag", {}).get("tasks", {}).items():
            display[(comp_name, task_name)] = (task["componentRef"]["name"], task["taskInfo"]["name"])

    details = {t.task_id: t for t in job.task_details}

    def entry_of(task):
        # task names repeat across branches (model-batch-predict), so look the task up in its enclosing component
        parent = details.get(task.parent_task_id)
        seen = 0
        while seen < 10:
            holder = f"comp-{parent.task_name}" if parent is not None else "root"
            if (holder, task.task_name) in display:
                return display[(holder, task.task_name)]
            if parent is None:
                return None
            parent = details.get(parent.parent_task_id)
            seen += 1
        return None

    def branch_of(task):
        seen = 0
        while task is not None and seen < 10:
            if task.task_name in branches:
                return branches[task.task_name]
            task = details.get(task.parent_task_id)
            seen += 1
        return None

    state_map = {"SUCCEEDED": "SUCCEEDED", "SKIPPED": "SKIPPED", "NOT_TRIGGERED": "SKIPPED"}
    steps = []
    for task in details.values():
        entry = entry_of(task)
        if task.task_name.startswith(("condition", "exit-handler")) or task.task_name in helpers:
            continue
        if (entry and entry[0] in wrappers) or task.task_name in branches:
            continue
        name = (entry[1] if entry else task.task_name).replace("-", "_").replace(" ", "_")
        started, ended = task.start_time, task.end_time
        seconds = (ended - started).total_seconds() if started and ended else None
        steps.append(
            {
                "run_id": run_id,
                "step": name,
                "branch": branch_of(task) or "shared",
                "status": state_map.get(task.state.name, "FAILED" if task.state.name != "SUCCEEDED" else "SUCCEEDED"),
                "started_at": iso(started),
                "ended_at": iso(ended),
                "duration_seconds": seconds,
            }
        )
    steps.sort(key=lambda s: (s["started_at"] or "~", s["step"]))

    final = "SUCCEEDED" if status.state == "SUCCEEDED" else "FAILED"
    started = job.gca_resource.start_time or job.gca_resource.create_time
    console = (
        f"https://console.cloud.google.com/vertex-ai/locations/{location}/pipelines/runs/{run_id}?project={project}"
    )
    parameters = {"project": project, "location": location, "run_date": run_date, "train_automl": train_automl}
    run = {
        "run_id": run_id,
        "status": final,
        "started_at": iso(started),
        "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameters": parameters,
        "train_automl": train_automl,
        "data_as_of": run_date or iso(started)[:10],
        "console_url": console,
    }
    write("runs", [run])
    write("step_timings", steps)
    print(f"run {run_id}: {final}, {len(steps)} steps")
    return len(steps)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def publish_results(
    project: str,
    location: str,
    run_id: str,
    run_date: str,
    train_automl: bool,
    bq_dataset: str,
    dashboard_dataset: str,
    schema_json: str,
    churn_model_name: str,
    churn_version_id: str,
    churn_evaluation: Input[Artifact],
    churn_predictions: Input[Artifact],
    churn_explanations: Input[Artifact],
    clv_model_name: str,
    clv_version_id: str,
    clv_evaluation: Input[Artifact],
    demand_model: str,
    demand_training_seconds: float,
    demand_mape: float,
    demand_coverage: float,
    demand_series: int,
    demand_excluded_days: int,
    demand_daily: str,
    clv_train_table: str,
    demand_forecast_table: str,
    demand_holdout_metrics_table: str,
    demand_holdout_forecast_table: str,
    actual_days: int,
    clv_sample_rows: int,
    top_customers: int,
    churn_target: str,
    churn_negative_class: str,
    churn_positive_class: str,
) -> int:
    """Write the model, churn, CLV and demand tables for this run to the dashboard dataset."""
    if dashboard_dataset != "vertex_demo_dashboard":
        raise ValueError(f"publish writes only vertex_demo_dashboard, not {dashboard_dataset!r}")
    run_id = run_id.rsplit("/", 1)[-1]
    import csv
    import datetime
    import io
    import json
    import math
    import random

    from google.cloud import bigquery, storage

    schema = json.loads(schema_json)
    labels = {"app": "vertex-demo"}
    client = bigquery.Client(project=project)
    tables: dict[str, list] = {}

    def validate(table, rows):
        types = {f["name"]: f for f in schema[table]["fields"]}
        for row in rows:
            for name in row:
                if name not in types:
                    raise ValueError(f"{table}: column {name!r} is not in the publish schema")
            for name, field in types.items():
                value = row.get(name)
                if value is None:
                    if field["mode"] == "REQUIRED":
                        raise ValueError(f"{table}: required column {name!r} is missing")
                    continue
                kind = field["type"]
                ok = {
                    "STRING": isinstance(value, str),
                    "INT64": isinstance(value, int) and not isinstance(value, bool),
                    "FLOAT64": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                    "BOOL": isinstance(value, bool),
                    "TIMESTAMP": isinstance(value, str),
                    "DATE": isinstance(value, str),
                    "JSON": isinstance(value, (dict, list, str)),
                }[kind]
                if not ok:
                    raise ValueError(f"{table}.{name}: {value!r} is not a {kind}")

    def write(table, rows):
        validate(table, rows)
        fields = [
            bigquery.SchemaField(f["name"], f["type"], mode=f["mode"], description=f["description"])
            for f in schema[table]["fields"]
        ]
        ref = f"{project}.{dashboard_dataset}.{table}"
        created = bigquery.Table(ref, schema=fields)
        created.labels = labels
        created.description = schema[table]["description"]
        client.create_table(created, exists_ok=True)
        config = bigquery.QueryJobConfig(
            labels=labels, query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
        )
        client.query(f"DELETE FROM `{ref}` WHERE run_id = @run_id", job_config=config).result()
        if rows:
            json_columns = {f["name"] for f in schema[table]["fields"] if f["type"] == "JSON"}
            # the client serializes each row once; a JSON column must hold the object, or it is stored as a string of it
            body = [
                {k: (json.loads(v) if k in json_columns and isinstance(v, str) else v) for k, v in r.items()}
                for r in rows
            ]
            load = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                schema=fields,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                labels=labels,
            )
            client.load_table_from_json(body, ref, job_config=load).result()

    def query(sql):
        config = bigquery.QueryJobConfig(labels=labels)
        return [dict(r.items()) for r in client.query(sql, job_config=config).result()]

    def num(value):
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def whole(value):
        return None if value is None else int(float(value))

    def count(value):
        return 0 if value is None else int(float(value))

    def day(value):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def read_text(uri):
        bucket, key = uri[5:].split("/", 1)
        return storage.Client(project=project).bucket(bucket).blob(key).download_as_text()

    def console_model(name, version):
        model_id = name.split("/")[-1]
        return (
            f"https://console.cloud.google.com/vertex-ai/models/locations/{location}/models/{model_id}"
            f"/versions/{version}?project={project}"
        )

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # --- churn -------------------------------------------------------------------------------------------
    with open(churn_evaluation.path) as f:
        churn = json.load(f)
    training = churn.get("training") or {}
    confusion = [
        {
            "run_id": run_id,
            "threshold": num(p["threshold"]),
            "tp": count(p["tp"]),
            "fp": count(p["fp"]),
            "tn": count(p["tn"]),
            "fn": count(p["fn"]),
        }
        for p in churn.get("confusion", [])
        if p.get("threshold") is not None
    ]
    curves = [
        {
            "run_id": run_id,
            "curve": "pr",
            "threshold": num(p["threshold"]),
            "x": num(p["recall"]),
            "y": num(p["precision"]),
        }
        for p in churn.get("pr_curve", [])
        if p.get("threshold") is not None and p.get("recall") is not None and p.get("precision") is not None
    ] + [
        {"run_id": run_id, "curve": "roc", "threshold": num(p["threshold"]), "x": num(p["fpr"]), "y": num(p["tpr"])}
        for p in churn.get("roc_curve", [])
        if p.get("threshold") is not None and p.get("fpr") is not None and p.get("tpr") is not None
    ]
    importance = sorted(
        (
            {"feature": i["feature"], "importance": abs(float(i["importance"]))}
            for i in churn.get("feature_importance", [])
        ),
        key=lambda i: -i["importance"],
    )
    importance = [{"run_id": run_id, **i, "rank": n + 1} for n, i in enumerate(importance)]
    first = confusion[0] if confusion else None
    total = first and first["tp"] + first["fp"] + first["tn"] + first["fn"]
    positives = first and first["tp"] + first["fn"]
    churn_metrics = {
        "auc_pr": churn.get("auprc"),
        "auc_roc": churn.get("auroc"),
        "log_loss": churn.get("log_loss"),
        "positive_rate": (positives / total) if total else None,
        "test_customers": float(total) if total else None,
    }
    if churn.get("operating_point"):
        op = churn["operating_point"]
        churn_metrics.update(
            {
                "precision": op.get("precision"),
                "recall": op.get("recall"),
                "f1": op.get("f1"),
                "threshold": op.get("threshold"),
            }
        )
    if not confusion or not curves or not importance or churn_metrics["auc_pr"] is None:
        raise RuntimeError(
            "churn evaluation is incomplete: "
            f"{len(confusion)} confusion rows, {len(curves)} curve points, {len(importance)} importances, "
            f"auc_pr={churn_metrics['auc_pr']}, importance error={churn.get('feature_importance_error')}"
        )
    budget = training.get("budget_milli_node_hours")
    tables["churn_confusion"] = confusion
    tables["churn_curve_points"] = curves
    tables["churn_feature_importance"] = importance
    models = [
        {
            "run_id": run_id,
            "model": "churn",
            "training_method": "AutoML Tabular",
            "model_resource_name": churn_model_name,
            "model_version": churn_version_id,
            "trained_at": training.get("trained_at"),
            "headline_metric_name": f"AUC-PR ({churn.get('positive_class', churn_positive_class)} class)",
            "headline_metric_value": num(churn_metrics["auc_pr"]),
            "console_url": console_model(churn_model_name, churn_version_id),
            "machine_type": "AutoML managed",
            "budget": f"{int(float(budget)):,} milli node hours" if budget else None,
            "trial_count": None,
            "training_duration_seconds": num(training.get("duration_seconds")),
            "trained_in_run": bool(train_automl and churn.get("gate", {}).get("promoted", True)),
        }
    ]
    metrics = [
        {"run_id": run_id, "model": "churn", "metric": k, "value": num(v)}
        for k, v in churn_metrics.items()
        if v is not None
    ]

    # Top at-risk customers: highest score for the churned class, with the top 3 attributions of that class.
    def table_of(artifact):
        m = artifact.metadata
        return f"`{m['projectId']}.{m['datasetId']}.{m['tableId']}`"

    pred_col = f"predicted_{churn_target}"
    rows = query(
        f"""WITH scored AS (
  SELECT user_id,
    (SELECT s FROM UNNEST({pred_col}.scores) AS s WITH OFFSET i
     WHERE {pred_col}.classes[OFFSET(i)] = '{churn_negative_class}') AS churn_score
  FROM {table_of(churn_predictions)}
)
SELECT SAFE_CAST(p.user_id AS INT64) AS user_id, p.churn_score, TO_JSON_STRING(e.explanation) AS explanation
FROM scored AS p JOIN {table_of(churn_explanations)} AS e USING (user_id)
ORDER BY p.churn_score DESC, user_id
LIMIT {int(top_customers)}"""
    )

    def attributions(raw):
        """Top 3 (feature, value) pairs for the churned class from the explanation column."""
        if not raw:
            return []
        data = json.loads(raw) if isinstance(raw, str) else raw
        found = data.get("attributions", []) if isinstance(data, dict) else []
        chosen = None
        for item in found:
            if item.get("outputDisplayName") == churn_negative_class:
                chosen = item
                break
        chosen = chosen or (found[0] if found else {})
        feats = chosen.get("featureAttributions") or {}
        if isinstance(feats, str):
            feats = json.loads(feats)
        ranked = sorted(feats.items(), key=lambda kv: -abs(float(kv[1])))
        return [(k, float(v)) for k, v in ranked[:3]]

    customers = []
    for rank, row in enumerate(rows, start=1):
        record = {
            "run_id": run_id,
            "user_id": whole(row["user_id"]),
            "churn_score": num(row["churn_score"]),
            "rank": rank,
        }
        for n, (feature, value) in enumerate(attributions(row["explanation"]), start=1):
            record[f"reason_{n}_feature"] = feature
            record[f"reason_{n}_value"] = value
        customers.append(record)
    if not customers:
        raise RuntimeError("no churn predictions with explanations found")
    tables["churn_top_customers"] = customers

    # --- CLV ---------------------------------------------------------------------------------------------
    with open(clv_evaluation.path) as f:
        clv = json.load(f)
    clv_m = clv["metrics"]
    best = str(clv["best_trial_id"])
    tables["clv_tuning_trials"] = [
        {
            "run_id": run_id,
            "trial_id": str(t["trial_id"]),
            "hyperparameters": t.get("hyperparameters") or {},
            "rmse": num(t.get("rmse")),
            "is_best": str(t["trial_id"]) == best,
        }
        for t in clv["trials"]
    ]
    frequency = {int(r["user_id"]): r["frequency"] for r in query(f"SELECT user_id, frequency FROM {clv_train_table}")}

    def segment(user_id):
        n = frequency.get(user_id) or 0
        return "One-time" if n <= 1 else "Occasional" if n == 2 else "Regular" if n <= 4 else "High value"

    scored = []
    for r in csv.DictReader(io.StringIO(read_text(clv["validation_predictions"]))):
        if r["predicted"] in ("", None) or r["actual"] in ("", None):
            continue
        scored.append((int(float(r["user_id"])), float(r["predicted"]), float(r["actual"])))
    if not scored:
        raise RuntimeError("no CLV validation predictions found")
    by_segment: dict[str, list] = {}
    for user_id, predicted, actual in scored:
        by_segment.setdefault(segment(user_id), []).append((predicted, actual))
    tables["clv_error_by_segment"] = [
        {
            "run_id": run_id,
            "segment": name,
            "customers": len(pairs),
            "mae": sum(abs(p - a) for p, a in pairs) / len(pairs),
            "rmse": math.sqrt(sum((p - a) ** 2 for p, a in pairs) / len(pairs)),
            "bias": sum(p - a for p, a in pairs) / len(pairs),
        }
        for name, pairs in sorted(by_segment.items())
    ]
    ordered = sorted(scored, key=lambda t: -t[1])
    overall_actual = sum(a for _, _, a in scored) / len(scored)
    deciles = []
    size = len(ordered) / 10
    for d in range(10):
        chunk = ordered[round(d * size) : round((d + 1) * size)]
        mean_actual = sum(a for _, _, a in chunk) / len(chunk)
        deciles.append(
            {
                "run_id": run_id,
                "decile": d + 1,
                "customers": len(chunk),
                "predicted_revenue": sum(p for _, p, _ in chunk),
                "actual_revenue": sum(a for _, _, a in chunk),
                "lift": mean_actual / overall_actual if overall_actual else None,
            }
        )
    tables["clv_deciles"] = deciles
    sample = random.Random(42).sample(scored, min(int(clv_sample_rows), len(scored)))
    tables["clv_predictions_sample"] = [
        {"run_id": run_id, "user_id": u, "predicted": p, "actual": a, "segment": segment(u)} for u, p, a in sample
    ]
    models.append(
        {
            "run_id": run_id,
            "model": "clv",
            "training_method": "Custom XGBoost + hyperparameter tuning",
            "model_resource_name": clv_model_name,
            "model_version": clv_version_id,
            "trained_at": clv["trials"][-1].get("end_time") if clv.get("trials") else now,
            "headline_metric_name": "RMSE",
            "headline_metric_value": num(clv_m["rmse"]),
            "console_url": console_model(clv_model_name, clv_version_id),
            "machine_type": f"{clv.get('machine_type', 'n1-standard-4')} (CPU)",
            "budget": f"{clv.get('max_trial_count', len(clv['trials']))} trials, 3 in parallel",
            "trial_count": len(clv["trials"]),
            "training_duration_seconds": num(clv.get("duration_seconds")),
            "trained_in_run": True,
        }
    )
    metrics += [
        {"run_id": run_id, "model": "clv", "metric": k, "value": num(clv_m.get(k))}
        for k in ("rmse", "mae", "r2", "baseline_rmse")
        if clv_m.get(k) is not None
    ] + [{"run_id": run_id, "model": "clv", "metric": "test_customers", "value": float(len(scored))}]

    # --- demand ------------------------------------------------------------------------------------------
    holdout = query(f"SELECT * FROM {demand_holdout_forecast_table}")
    forecast = query(f"SELECT * FROM {demand_forecast_table}")
    first_holdout = min(day(r["date"]) for r in holdout)
    actuals = query(
        f"SELECT category, sale_date, units FROM {demand_daily} "
        f"WHERE sale_date < DATE('{first_holdout}') "
        f"AND sale_date >= DATE_SUB(DATE('{first_holdout}'), INTERVAL {int(actual_days)} DAY)"
    )
    series_rows = (
        [
            {
                "run_id": run_id,
                "category": r["category"],
                "date": day(r["sale_date"]),
                "kind": "actual",
                "units": num(r["units"]),
            }
            for r in actuals
        ]
        + [
            {
                "run_id": run_id,
                "category": r["category"],
                "date": day(r["date"]),
                "kind": "holdout",
                "units": num(r["units"]),
                "forecast_units": num(r["forecast_units"]),
                "lower_bound": num(r["lower_bound"]),
                "upper_bound": num(r["upper_bound"]),
            }
            for r in holdout
        ]
        + [
            {
                "run_id": run_id,
                "category": r["category"],
                "date": day(r["date"]),
                "kind": "forecast",
                "forecast_units": num(r["forecast"]),
                "lower_bound": num(r["lower"]),
                "upper_bound": num(r["upper"]),
            }
            for r in forecast
        ]
    )
    tables["demand_forecast"] = series_rows
    holdout_metrics = query(f"SELECT * FROM {demand_holdout_metrics_table} WHERE category != 'ALL'")
    tables["demand_holdout_metrics"] = [
        {
            "run_id": run_id,
            "category": r["category"],
            "mape": None if r["mape"] is None else num(r["mape"]) * 100,
            "coverage": num(r["coverage"]),
            "holdout_days": whole(r["holdout_days"]),
        }
        for r in holdout_metrics
    ]
    dataset_id = demand_model.strip("`").split(".")
    models.append(
        {
            "run_id": run_id,
            "model": "demand",
            "training_method": "BigQuery ML ARIMA_PLUS",
            "model_resource_name": demand_model.strip("`"),
            "model_version": None,
            "trained_at": now,
            "headline_metric_name": "MAPE",
            "headline_metric_value": num(demand_mape) * 100,
            "console_url": (
                f"https://console.cloud.google.com/bigquery?project={project}&p={dataset_id[0]}"
                f"&d={dataset_id[1]}&m={dataset_id[2]}&page=model"
            ),
            "machine_type": "BigQuery ML managed",
            "budget": None,
            "trial_count": None,
            "training_duration_seconds": num(demand_training_seconds) if demand_training_seconds >= 0 else None,
            "trained_in_run": True,
        }
    )
    metrics += [
        {"run_id": run_id, "model": "demand", "metric": "mape_mean", "value": num(demand_mape) * 100},
        {"run_id": run_id, "model": "demand", "metric": "coverage_mean", "value": num(demand_coverage)},
        {"run_id": run_id, "model": "demand", "metric": "series", "value": float(demand_series)},
        {"run_id": run_id, "model": "demand", "metric": "excluded_trailing_days", "value": float(demand_excluded_days)},
    ]
    tables["models"] = models
    tables["model_metrics"] = metrics

    for name, rows_ in tables.items():
        validate(name, rows_)
    for name, rows_ in tables.items():
        write(name, rows_)
    counts = {name: len(rows_) for name, rows_ in tables.items()}
    print(json.dumps(counts))
    return sum(counts.values())
