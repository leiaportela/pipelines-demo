import csv
import io
import json
import types
from datetime import UTC, datetime, timedelta

import pytest
from google.cloud.aiplatform_v1.types.schedule import Schedule

from pipeline import publish, schedule
from pipeline import publish_schema as ps
from pipeline.compile import compile_pipeline
from pipeline.run import PROJECT

RUN_ID = "vertex-demo-test-20261126"
DAY = datetime(2026, 11, 20)


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def result(self):
        return self.rows


class FakeBigQuery:
    """Answers the publish step's reads from fixtures and records every write and every label."""

    def __init__(self, project=None):
        self.queries, self.loads, self.tables, self.label_sets = [], {}, [], []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        self.label_sets.append(job_config.labels)
        if "UNNEST" in sql:
            return Rows(
                [
                    {
                        "user_id": 100 + i,
                        "churn_score": 0.99 - i / 100,
                        "explanation": json.dumps(
                            {
                                "attributions": [
                                    {
                                        "outputDisplayName": "churned",
                                        "featureAttributions": {"recency": 0.4, "frequency": -0.2, "age": 0.1, "x": 0},
                                    }
                                ]
                            }
                        ),
                    }
                    for i in range(5)
                ]
            )
        if "clv_train" in sql:
            return Rows([{"user_id": u, "frequency": u % 6} for u in range(1, 41)])
        if "demand_holdout_forecast" in sql:
            return Rows(
                [
                    {
                        "category": c,
                        "date": DAY + timedelta(days=d),
                        "units": 5.0,
                        "forecast_units": 4.0,
                        "lower_bound": 1.0,
                        "upper_bound": 9.0,
                    }
                    for c in ("Jeans", "Socks")
                    for d in range(3)
                ]
            )
        if "demand_forecast" in sql:
            return Rows(
                [
                    {"category": c, "date": DAY + timedelta(days=10 + d), "forecast": 4.0, "lower": 1.0, "upper": 9.0}
                    for c in ("Jeans", "Socks")
                    for d in range(3)
                ]
            )
        if "demand_holdout_metrics" in sql:
            return Rows(
                [
                    {"category": "Jeans", "mape": 0.25, "coverage": 0.9, "holdout_days": 3},
                    {"category": "Socks", "mape": None, "coverage": 1.0, "holdout_days": 3},
                ]
            )
        if "demand_daily" in sql:
            return Rows(
                [{"category": "Jeans", "sale_date": DAY - timedelta(days=d), "units": 3.0} for d in range(1, 4)]
            )
        return Rows([])

    def create_table(self, table, exists_ok=False):
        self.tables.append(table)

    def load_table_from_json(self, body, ref, job_config=None):
        self.loads[ref.rsplit(".", 1)[1]] = body
        self.label_sets.append(job_config.labels)
        return Rows(None)


class FakeBlob:
    def download_as_text(self):
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["user_id", "actual", "predicted"])
        for u in range(1, 41):
            writer.writerow([u, u * 2.0, u * 2.0 + (1 if u % 2 else -1)])
        return out.getvalue()


class FakeStorage:
    def __init__(self, project=None):
        pass

    def bucket(self, name):
        return types.SimpleNamespace(blob=lambda key: FakeBlob())


CHURN = {
    "positive_class": "reorders",
    "auprc": 0.31,
    "auroc": 0.72,
    "log_loss": 0.25,
    "pr_curve": [{"threshold": t, "recall": 1 - t, "precision": 0.1 + t / 4} for t in (0.1, 0.5, 0.9)],
    "roc_curve": [{"threshold": t, "fpr": 1 - t, "tpr": 1 - t / 2} for t in (0.1, 0.5, 0.9)],
    "confusion": [{"threshold": t, "tp": 5, "fp": 20, "tn": 900, "fn": 10} for t in (0.1, 0.5, 0.9)],
    "operating_point": {"threshold": 0.5, "precision": 0.2, "recall": 0.4, "f1": 0.27},
    "feature_importance": [
        {"feature": "recency", "importance": -0.5, "rank": 1},
        {"feature": "age", "importance": 0.1},
    ],
    "training": {
        "duration_seconds": 3600.0,
        "budget_milli_node_hours": "1000",
        "trained_at": "2026-11-26T01:00:00+00:00",
    },
    "gate": {"promoted": True},
}
CLV = {
    "model_name": "projects/1/locations/us-central1/models/7",
    "version_id": "3",
    "best_trial_id": "2",
    "machine_type": "n1-standard-4",
    "duration_seconds": 2400.0,
    "max_trial_count": 3,
    "metrics": {"rmse": 41.0, "mae": 22.0, "r2": 0.2, "baseline_rmse": 45.0},
    "trials": [
        {
            "trial_id": str(i),
            "state": "SUCCEEDED",
            "hyperparameters": {"max_depth": 3 + i},
            "rmse": 40.0 + i,
            "end_time": "2026-11-26T02:00:00+00:00",
        }
        for i in (1, 2, 3)
    ],
    "validation_predictions": "gs://b/x/eval/validation_predictions.csv",
}


@pytest.fixture
def run_publish(monkeypatch, tmp_path):
    from google.cloud import bigquery, storage

    fake = FakeBigQuery()
    monkeypatch.setattr(bigquery, "Client", lambda project=None: fake)
    monkeypatch.setattr(storage, "Client", FakeStorage)

    def go(
        churn=CHURN, clv=CLV, train_automl=False, schema=None, run_id=RUN_ID, dashboard_dataset="vertex_demo_dashboard"
    ):
        paths = {}
        for name, body in (("churn", churn), ("clv", clv), ("pred", {}), ("expl", {})):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(body))
            paths[name] = types.SimpleNamespace(path=str(path), metadata={})
        paths["pred"].metadata = {"projectId": "p", "datasetId": "d", "tableId": "predictions"}
        paths["expl"].metadata = {"projectId": "p", "datasetId": "d", "tableId": "explained"}
        total = publish.publish_results.python_func(
            project="p",
            location="us-central1",
            run_id=run_id,
            run_date="2026-11-20",
            train_automl=train_automl,
            bq_dataset="vertex_demo",
            dashboard_dataset=dashboard_dataset,
            schema_json=schema or publish.schema_json(),
            churn_model_name="projects/1/locations/us-central1/models/9",
            churn_version_id="1",
            churn_evaluation=paths["churn"],
            churn_predictions=paths["pred"],
            churn_explanations=paths["expl"],
            clv_model_name=CLV["model_name"],
            clv_version_id="3",
            clv_evaluation=paths["clv"],
            demand_model="`p.vertex_demo.demand_model`",
            demand_training_seconds=120.0,
            demand_mape=0.3,
            demand_coverage=0.9,
            demand_series=2,
            demand_excluded_days=6,
            demand_daily="`p.vertex_demo.demand_daily`",
            clv_train_table="`p.vertex_demo.clv_train`",
            demand_forecast_table="`p.vertex_demo.demand_forecast`",
            demand_holdout_metrics_table="`p.vertex_demo.demand_holdout_metrics`",
            demand_holdout_forecast_table="`p.vertex_demo.demand_holdout_forecast`",
            actual_days=30,
            clv_sample_rows=10,
            top_customers=5,
            churn_target="outcome",
            churn_negative_class="churned",
            churn_positive_class="reorders",
        )
        return fake, total

    return go


PUBLISHED = set(ps.TABLES) - {"runs", "step_timings"}


@pytest.mark.parametrize("train_automl", [False, True])
def test_every_model_table_is_written_and_matches_the_schema(run_publish, train_automl):
    fake, total = run_publish(train_automl=train_automl)
    assert set(fake.loads) == PUBLISHED
    assert total == sum(len(v) for v in fake.loads.values())
    for table, rows in fake.loads.items():
        assert rows, table
        allowed = {f["name"] for f in ps.TABLES[table]["fields"]}
        assert all(set(r) <= allowed and r["run_id"] == RUN_ID for r in rows), table
    models = {r["model"]: r for r in fake.loads["models"]}
    assert set(models) == {"churn", "clv", "demand"}
    assert models["churn"]["trained_in_run"] is train_automl
    excluded = [r for r in fake.loads["model_metrics"] if r["metric"] == "excluded_trailing_days"]
    assert [(r["model"], r["value"]) for r in excluded] == [("demand", 6.0)]
    assert models["demand"]["headline_metric_value"] == pytest.approx(30.0)
    assert {r["curve"] for r in fake.loads["churn_curve_points"]} == {"pr", "roc"}
    assert sum(r["is_best"] for r in fake.loads["clv_tuning_trials"]) == 1
    assert {r["kind"] for r in fake.loads["demand_forecast"]} == {"actual", "holdout", "forecast"}
    top = fake.loads["churn_top_customers"][0]
    assert (top["rank"], top["reason_1_feature"]) == (1, "recency") and "reason_3_feature" in top
    assert fake.loads["clv_tuning_trials"][0]["hyperparameters"] == {"max_depth": 4}


def test_the_run_id_is_the_job_name_taken_from_the_resource_name(run_publish):
    fake, _ = run_publish(run_id="projects/1/locations/us-central1/pipelineJobs/" + RUN_ID)
    assert {r["run_id"] for rows in fake.loads.values() for r in rows} == {RUN_ID}


def test_the_pipeline_never_uses_the_billing_job_id_placeholder(spec):
    assert "pipeline_job_uuid" not in json.dumps(spec)


def test_json_columns_are_loaded_as_objects_not_strings(run_publish):
    fake, _ = run_publish()
    assert isinstance(fake.loads["clv_tuning_trials"][0]["hyperparameters"], dict)


def test_publish_refuses_any_dataset_but_the_dashboard(run_publish):
    with pytest.raises(ValueError, match="vertex_demo_dashboard"):
        run_publish(dashboard_dataset="vertex_demo")


def test_record_run_refuses_any_dataset_but_the_dashboard():
    with pytest.raises(ValueError, match="vertex_demo_dashboard"):
        publish.record_run.python_func(
            project="p",
            location="l",
            job_resource_name="x",
            run_id=RUN_ID,
            run_date="",
            train_automl=False,
            dashboard_dataset="vertex_demo",
            schema_json=publish.schema_json(),
            status=types.SimpleNamespace(state="FAILED"),
        )


def test_every_bigquery_job_carries_the_app_label_and_only_the_dashboard_is_written(run_publish):
    fake, _ = run_publish()
    assert fake.label_sets and all(labels == {"app": "vertex-demo"} for labels in fake.label_sets)
    assert all(t.labels == {"app": "vertex-demo"} for t in fake.tables)
    assert {t.dataset_id for t in fake.tables} == {"vertex_demo_dashboard"}
    deletes = [q for q in fake.queries if q.startswith("DELETE")]
    assert len(deletes) == len(PUBLISHED) and all("run_id = @run_id" in q for q in deletes)


def test_a_renamed_published_column_fails_the_publish(run_publish):
    schema = json.loads(publish.schema_json())
    for field in schema["clv_deciles"]["fields"]:
        if field["name"] == "lift":
            field["name"] = "lift_ratio"
    with pytest.raises(ValueError, match="lift"):
        run_publish(schema=json.dumps(schema))


def test_an_incomplete_churn_evaluation_fails(run_publish):
    with pytest.raises(RuntimeError, match="incomplete"):
        run_publish(churn={**CHURN, "feature_importance": []})


def test_automl_confusion_rows_with_string_and_missing_counts_are_published(run_publish):
    edge = [
        {"threshold": -0.005, "tp": "428", "fp": "5802", "tn": None, "fn": None},
        {"threshold": None, "tp": "428", "fp": "5802", "tn": None, "fn": None},
        {"threshold": 0.5, "tp": "10", "fp": "20", "tn": "5782", "fn": "418"},
        {"threshold": 1.005, "tp": None, "fp": None, "tn": "5802", "fn": "428"},
    ]
    fake, _ = run_publish(churn={**CHURN, "confusion": edge})
    rows = fake.loads["churn_confusion"]
    assert [r["threshold"] for r in rows] == [-0.005, 0.5, 1.005]
    assert rows[0]["tn"] == 0 and rows[0]["fn"] == 0 and rows[0]["tp"] == 428
    assert rows[2]["tp"] == 0 and rows[2]["fp"] == 0


class Task:
    def __init__(self, task_id, name, parent, state="SUCCEEDED", seconds=60):
        self.task_id, self.task_name, self.parent_task_id = task_id, name, parent
        self.state = types.SimpleNamespace(name=state)
        self.start_time = datetime(2026, 11, 26, 6, tzinfo=UTC)
        self.end_time = self.start_time + timedelta(seconds=seconds)


def test_record_run_writes_the_final_state_of_a_failed_run(monkeypatch, tmp_path):
    from google.cloud import aiplatform, bigquery
    from google.protobuf import json_format, struct_pb2

    spec_path = compile_pipeline(tmp_path / "pipeline.json")
    pb = struct_pb2.Struct()
    json_format.ParseDict(json.loads(spec_path.read_text()), pb)
    fake = FakeBigQuery()
    monkeypatch.setattr(bigquery, "Client", lambda project=None: fake)
    job = types.SimpleNamespace(
        gca_resource=types.SimpleNamespace(
            _pb=types.SimpleNamespace(pipeline_spec=pb),
            start_time=datetime(2026, 11, 26, 6, tzinfo=UTC),
            create_time=datetime(2026, 11, 26, 6, tzinfo=UTC),
        ),
        task_details=[
            Task(1, "exit-handler-1", None),
            Task(2, "demand-bqml-arima", 1),
            Task(3, "bigquerycreatemodeljob", 2),
            Task(4, "model-batch-predict", 5, state="FAILED"),
            Task(5, "clv-xgboost", 1),
            Task(6, "record-run", None, state="RUNNING"),
        ],
    )
    monkeypatch.setattr(aiplatform, "init", lambda **kw: None)
    monkeypatch.setattr(aiplatform.PipelineJob, "get", staticmethod(lambda name: job))
    publish.record_run.python_func(
        project="p",
        location="us-central1",
        job_resource_name="projects/1/locations/us-central1/pipelineJobs/x",
        run_id=RUN_ID,
        run_date="",
        train_automl=False,
        dashboard_dataset="vertex_demo_dashboard",
        schema_json=publish.schema_json(),
        status=types.SimpleNamespace(state="FAILED"),
    )
    assert set(fake.loads) == {"runs", "step_timings"}
    (run,) = fake.loads["runs"]
    assert run["status"] == "FAILED" and run["data_as_of"] == "2026-11-26" and run["train_automl"] is False
    steps = {s["step"]: s for s in fake.loads["step_timings"]}
    assert steps["model_batch_predict"]["status"] == "FAILED" and steps["model_batch_predict"]["branch"] == "clv"
    assert all(s["step"] != "record_run" and s["duration_seconds"] == 60 for s in steps.values())
    assert all(labels == {"app": "vertex-demo"} for labels in fake.label_sets)


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    return json.loads(compile_pipeline(tmp_path_factory.mktemp("c") / "pipeline.json").read_text())


def test_the_exit_handler_wraps_every_branch_and_publish(spec):
    root = spec["root"]["dag"]["tasks"]
    (handler,) = [n for n, t in root.items() if n.startswith("exit-handler")]
    exit_task = root["record-run"]
    assert exit_task["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED"
    assert handler in exit_task["dependentTasks"]
    inner = spec["components"][f"comp-{handler}"]["dag"]["tasks"]
    assert {"build-features", "churn-automl", "clv-xgboost", "demand-bqml-arima", "publish-results"} <= set(inner)
    assert set(inner["publish-results"]["dependentTasks"]) >= {"churn-automl", "clv-xgboost", "demand-bqml-arima"}
    assert "record-run" not in inner


def test_the_publish_schema_is_the_contract(spec):
    constants = [
        t["inputs"]["parameters"]["schema_json"]["runtimeValue"]["constant"]
        for t in spec["root"]["dag"]["tasks"].values()
        if "schema_json" in t.get("inputs", {}).get("parameters", {})
    ]
    assert constants and all(json.loads(c) == ps.TABLES for c in constants)


def test_the_schedule_is_weekly_labeled_and_never_trains_automl():
    s = schedule.schedule_spec()
    assert s["parameter_values"]["train_automl"] is False
    assert s["labels"] == {"app": "vertex-demo"}
    assert s["cron"] == "TZ=America/Sao_Paulo 0 6 * * 1"
    assert s["max_concurrent_run_count"] == 1
    assert s["service_account"].startswith("vertex-demo-pipelines@")
    assert s["pipeline_root"].startswith(f"gs://{PROJECT}-vertex-demo/")


ACTIVE = types.SimpleNamespace(state=Schedule.State.ACTIVE)  # the real enum, as Vertex returns it


def test_the_schedule_is_created_once(monkeypatch):
    from google.cloud import aiplatform

    created = []
    existing = types.SimpleNamespace(gca_resource=ACTIVE, resource_name="projects/1/schedules/2")
    monkeypatch.setattr(schedule, "is_current", lambda s: True)
    monkeypatch.setattr(aiplatform, "init", lambda **kw: None)
    monkeypatch.setattr(aiplatform.PipelineJobSchedule, "list", staticmethod(lambda filter=None: [existing]))
    monkeypatch.setattr(aiplatform.PipelineJobSchedule, "create", lambda self, **kw: created.append(kw), raising=False)
    assert schedule.ensure() is existing and not created


def test_a_missing_schedule_is_created_with_the_spec(monkeypatch):
    from google.cloud import aiplatform

    seen = {}

    class Job:
        def __init__(self, **kw):
            seen["job"] = kw

    class Schedule:
        gca_resource = ACTIVE

        def __init__(self, pipeline_job, display_name):
            seen["name"] = display_name

        def create(self, **kw):
            seen["create"] = kw

    monkeypatch.setattr(aiplatform, "init", lambda **kw: None)
    monkeypatch.setattr(aiplatform, "PipelineJob", Job)
    monkeypatch.setattr(
        aiplatform, "PipelineJobSchedule", type("S", (Schedule,), {"list": staticmethod(lambda filter=None: [])})
    )
    schedule.ensure()
    assert seen["job"]["parameter_values"]["train_automl"] is False
    assert seen["job"]["labels"] == {"app": "vertex-demo"}
    assert seen["create"]["cron"] == "TZ=America/Sao_Paulo 0 6 * * 1"
    assert seen["create"]["max_concurrent_run_count"] == 1


def test_a_stale_schedule_is_deleted_and_created_again(monkeypatch):
    from google.cloud import aiplatform

    events = []

    class Old:
        gca_resource = ACTIVE
        resource_name = "projects/1/schedules/old"

        def delete(self):
            events.append("delete")

    class Job:
        def __init__(self, **kw):
            pass

    class New:
        gca_resource = ACTIVE
        resource_name = "projects/1/schedules/new"

        def __init__(self, pipeline_job, display_name):
            pass

        def create(self, **kw):
            events.append("create")

        @staticmethod
        def list(filter=None):
            return [Old()]

    monkeypatch.setattr(schedule, "is_current", lambda s: False)
    monkeypatch.setattr(aiplatform, "init", lambda **kw: None)
    monkeypatch.setattr(aiplatform, "PipelineJob", Job)
    monkeypatch.setattr(aiplatform, "PipelineJobSchedule", New)
    schedule.ensure()
    assert events == ["delete", "create"]


def test_a_schedule_is_current_only_when_spec_cron_and_parameters_match(tmp_path, monkeypatch):
    from google.cloud.aiplatform_v1.types import pipeline_job
    from google.cloud.aiplatform_v1.types import schedule as schedule_types
    from google.protobuf import json_format, struct_pb2

    from pipeline.compile import compile_pipeline

    monkeypatch.setattr(schedule, "COMPILED", compile_pipeline(tmp_path / "pipeline.json"))
    compiled = json.loads(schedule.COMPILED.read_text())

    def live(spec, cron=None, params=None):
        pb = struct_pb2.Struct()
        json_format.ParseDict(spec, pb)
        job = pipeline_job.PipelineJob(pipeline_spec=pb)
        json_format.ParseDict({"parameterValues": params or schedule.PARAMETERS}, job._pb.runtime_config)
        resource = schedule_types.Schedule(
            cron=cron or schedule.schedule_spec()["cron"], max_concurrent_run_count=schedule.MAX_CONCURRENT_RUNS
        )
        resource.create_pipeline_job_request.pipeline_job = job
        return types.SimpleNamespace(gca_resource=resource)

    assert schedule.is_current(live(compiled))
    cached = json.loads(json.dumps(compiled))
    for comp in cached["components"].values():
        for task in comp.get("dag", {}).get("tasks", {}).values():
            task["cachingOptions"] = {}
    assert schedule.is_current(live(cached))
    assert not schedule.is_current(live({**compiled, "schemaVersion": "0"}))
    assert not schedule.is_current(live(compiled, cron="0 7 * * 1"))
    assert not schedule.is_current(live(compiled, params={**schedule.PARAMETERS, "train_automl": True}))


@pytest.mark.parametrize("argv", [["--help"], ["--bogus"], ["extra"]])
def test_schedule_help_and_unknown_arguments_have_no_side_effects(monkeypatch, argv):
    monkeypatch.setattr(schedule, "ensure", lambda: pytest.fail("ensure must not run"))
    with pytest.raises(SystemExit):
        schedule.main(argv)
