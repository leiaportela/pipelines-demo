"""Shapes BigQuery hands back that the fixtures don't show: numbers as strings and Decimals, NULLs,
tables that don't exist yet, an empty dataset and a history with failed runs only."""
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.backend import data
from app.backend.main import create_app

PAGES = ["/api/overview", "/api/churn", "/api/clv", "/api/demand", "/api/runs/latest"]
ALL_TABLES = sorted(data.TABLES)


@pytest.fixture
def client(fake_bigquery):
    return TestClient(create_app())


@pytest.mark.parametrize("path", PAGES)
def test_missing_tables_are_an_empty_state_not_an_error(client, fake_bigquery, path):
    fake_bigquery.missing = set(ALL_TABLES)
    r = client.get(path)
    assert r.status_code == 200
    assert r.json()["empty"] is True and r.json()["run"] is None


@pytest.mark.parametrize("path", PAGES)
def test_empty_tables_are_an_empty_state(client, fake_bigquery, path):
    fake_bigquery.tables = {t: [] for t in ALL_TABLES}
    r = client.get(path)
    assert r.status_code == 200 and r.json()["empty"] is True


def test_only_failed_runs_is_an_empty_state_with_history(client, fake_bigquery):
    failed = [dict(r, status="FAILED") for r in data._fixture_table("runs")]
    fake_bigquery.tables = {"runs": failed}
    body = client.get("/api/overview").json()
    assert body["empty"] is True
    assert [r["status"] for r in body["history"]] == ["FAILED"] * len(failed)
    assert client.get("/api/runs").json()["runs"]


def test_unknown_run_is_still_a_404(client):
    assert client.get("/api/churn?run_id=nope").status_code == 404


def test_other_errors_are_not_swallowed(client, fake_bigquery, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(fake_bigquery.Client, "query", boom)
    with pytest.raises(RuntimeError):
        client.get("/api/runs")


def test_decimals_strings_and_nulls_pass_through(client, fake_bigquery):
    runs = [dict(r) for r in data._fixture_table("runs")]
    metrics = []
    for r in data._fixture_table("model_metrics"):
        r = dict(r)
        r["value"] = Decimal(str(r["value"]))  # NUMERIC
        metrics.append(r)
    models = [dict(r, headline_metric_value=None) for r in data._fixture_table("models")]  # NULL
    steps = [dict(r, duration_seconds=str(r["duration_seconds"])) for r in data._fixture_table("step_timings")]
    trials = [dict(r, hyperparameters='{"max_depth": 6}') for r in data._fixture_table("clv_tuning_trials")]
    fake_bigquery.tables = {
        "runs": runs, "model_metrics": metrics, "models": models, "step_timings": steps, "clv_tuning_trials": trials,
    }
    ov = client.get("/api/overview")
    assert ov.status_code == 200
    body = ov.json()
    assert body["run"]["run_id"]
    assert all(m["headline_metric_value"] is None for m in body["models"].values())
    assert all(isinstance(v, float) for m in body["models"].values() for v in m["metrics"].values())
    assert all(isinstance(s["duration_seconds"], str) for s in body["steps"])  # frontend n() reads these
    clv = client.get("/api/clv").json()
    assert clv["trials"][0]["hyperparameters"] == {"max_depth": 6}


def test_pipeline_root_task_is_not_a_step_and_double_encoded_json_is_decoded(client, fake_bigquery):
    steps = [dict(r) for r in data._fixture_table("step_timings")]
    for run in data._fixture_table("runs"):  # the root task of every run, whichever is the latest
        steps.append({"run_id": run["run_id"], "step": "vertex_demo_e2e_2_20261126", "branch": "shared",
                      "status": "FAILED", "started_at": run["started_at"], "ended_at": None,
                      "duration_seconds": None})
    runs = [dict(r, parameters='"{\\"train_automl\\": false}"') for r in data._fixture_table("runs")]
    fake_bigquery.tables = {"step_timings": steps, "runs": runs}
    body = client.get("/api/overview").json()
    assert not [s for s in body["steps"] if s["step"].startswith("vertex_demo")]
    assert body["run"]["parameters"] == {"train_automl": False}


def test_deciles_follow_the_contract_numbering(client):
    deciles = client.get("/api/clv").json()["deciles"]
    top = next(r for r in deciles if r["decile"] == 1)
    bottom = next(r for r in deciles if r["decile"] == 10)
    assert top["predicted_revenue"] > bottom["predicted_revenue"]  # 1 is the highest predicted spend


def test_stale_rows_are_served_while_one_refresh_runs(fake_bigquery, monkeypatch):
    import time

    monkeypatch.setattr(data, "CACHE_TTL", 0.0)
    first = data.rows("runs")
    n = len(fake_bigquery.jobs)
    assert data.rows("runs") == first  # expired: stale rows come back at once
    deadline = time.time() + 2
    while len(fake_bigquery.jobs) == n and time.time() < deadline:
        time.sleep(0.01)
    assert len(fake_bigquery.jobs) == n + 1  # exactly one background refresh


def test_page_tables_are_read_in_one_round(client, fake_bigquery):
    client.get("/api/churn")
    reads = [sql for sql, _ in fake_bigquery.jobs]
    assert len(reads) == len(set(reads))  # no table is queried twice for one page
