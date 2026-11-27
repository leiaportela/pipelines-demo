import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backend import data
from app.backend.main import create_app

SCHEMA = json.loads(Path("pipeline/publish-schema.json").read_text())
LATEST = "run-20261125-0600"


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_route_answers_json_not_the_spa(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert "text/html" not in r.headers["content-type"]


def test_unknown_api_is_404(client):
    assert client.get("/api/nope").status_code == 404


def test_latest_run_is_the_latest_successful(client):
    body = client.get("/api/runs/latest").json()
    assert body["run_id"] == LATEST and body["status"] == "SUCCEEDED"


def test_history_includes_the_failed_run(client):
    runs = client.get("/api/runs").json()["runs"]
    assert [r["status"] for r in runs] == ["SUCCEEDED", "FAILED", "SUCCEEDED", "SUCCEEDED"]


def test_unknown_run_is_404(client):
    assert client.get("/api/overview?run_id=nope").status_code == 404


def test_overview(client):
    body = client.get("/api/overview").json()
    assert body["run"]["run_id"] == LATEST
    assert {s["branch"] for s in body["steps"]} == {"shared", "churn", "clv", "demand"}
    assert set(body["models"]) == {"churn", "clv", "demand"}
    assert body["models"]["churn"]["metrics"]["auc_pr"] > 0
    assert len(body["history"]) == 4


def test_churn(client):
    body = client.get("/api/churn").json()
    assert len(body["curves"]["pr"]) > 20 and len(body["curves"]["roc"]) > 20
    assert len(body["confusion"]) == 19
    row = body["confusion"][9]
    assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == 18240
    assert [c["rank"] for c in body["customers"]] == sorted(c["rank"] for c in body["customers"])
    assert body["importance"][0]["rank"] == 1


def test_clv(client):
    body = client.get("/api/clv").json()
    assert len(body["trials"]) == 12 and sum(t["is_best"] for t in body["trials"]) == 1
    assert body["trials"][0]["hyperparameters"]["max_depth"]
    assert len(body["deciles"]) == 10 and len(body["segments"]) == 4
    best = next(t for t in body["trials"] if t["is_best"])
    assert best["rmse"] == min(t["rmse"] for t in body["trials"])


def test_demand(client):
    body = client.get("/api/demand").json()
    assert len(body["series"]) == 26 and len(body["holdout"]) == 26
    first = next(iter(body["series"].values()))
    assert body["columns"][0] == "date" and first[-1][2] is not None and first[-1][3] <= first[-1][2] <= first[-1][4]


def test_run_without_detail_rows_is_empty_not_an_error(client):
    body = client.get("/api/churn?run_id=run-20261104-0600").json()
    assert body["model"]["model"] == "churn" and body["curves"]["pr"] == []


def test_fixture_mode_latency(client):
    import time

    client.get("/api/demand")
    t = time.perf_counter()
    for _ in range(20):
        assert client.get("/api/demand").status_code == 200
    assert (time.perf_counter() - t) / 20 < 0.3


def test_dataset_guard_refuses_other_datasets(monkeypatch):
    monkeypatch.setenv("BQ_DATASET", "vertex_demo_features")
    with pytest.raises(data.DatasetError):
        data.dataset()


def test_dataset_guard_surfaces_as_error(client, monkeypatch):
    monkeypatch.setenv("DASHBOARD_FIXTURES", "0")
    monkeypatch.setenv("BQ_DATASET", "other")
    assert client.get("/api/overview").status_code == 500


def test_dataset_guard_applies_in_fixture_mode(client, monkeypatch):
    monkeypatch.setenv("BQ_DATASET", "other")
    assert client.get("/api/overview").status_code == 500


def test_table_qualifies_allowed_dataset_and_refuses_unknown_tables(monkeypatch):
    monkeypatch.setenv("BQ_PROJECT", "p")
    monkeypatch.delenv("BQ_DATASET", raising=False)
    assert data.table("runs") == "`p.vertex_demo_dashboard.runs`"
    for bad in ("runs; DROP", "other_table", "information_schema"):
        with pytest.raises(data.DatasetError):
            data.table(bad)


def test_table_allowlist_matches_the_contract():
    assert data.TABLES == set(SCHEMA["tables"])


def test_bigquery_path_is_parameterized_and_guarded(monkeypatch):
    seen = []

    def fake_query(sql, params=None):
        seen.append((sql, params))
        started = datetime(2026, 11, 25, tzinfo=UTC)
        if "runs" in sql:
            return [{"run_id": "r1", "status": "SUCCEEDED", "started_at": started, "ended_at": None}]
        return []

    monkeypatch.setenv("BQ_PROJECT", "p")
    monkeypatch.delenv("BQ_DATASET", raising=False)
    monkeypatch.delenv("DASHBOARD_FIXTURES", raising=False)
    monkeypatch.setattr(data, "query", fake_query)
    for path in ("overview", "churn", "clv", "demand"):
        assert TestClient(create_app()).get(f"/api/{path}").status_code == 200
    assert seen
    for sql, params in seen:
        assert re.search(r"FROM `p\.vertex_demo_dashboard\.\w+`", sql)
        assert "r1" not in sql
        assert params in (None, {"run_id": "r1"})
    tables = {re.search(r"vertex_demo_dashboard\.(\w+)", s).group(1) for s, _ in seen}
    assert tables <= set(SCHEMA["tables"])


def test_every_fixture_row_matches_the_contract():
    types = {
        "STRING": str, "INT64": int, "FLOAT64": (int, float), "BOOL": bool, "TIMESTAMP": str, "DATE": str,
        "JSON": (dict, list),
    }
    fixtures = Path("app/backend/fixtures")
    assert {p.stem for p in fixtures.glob("*.json")} == set(SCHEMA["tables"])
    for name, spec in SCHEMA["tables"].items():
        fields = {f["name"]: f for f in spec["fields"]}
        rows = json.loads((fixtures / f"{name}.json").read_text())
        assert rows, name
        for row in rows:
            assert set(row) == set(fields), name
            for col, value in row.items():
                if value is None:
                    assert fields[col]["mode"] != "REQUIRED", (name, col)
                    continue
                ok = isinstance(value, types[fields[col]["type"]])
                if fields[col]["type"] in ("INT64", "FLOAT64"):
                    ok = ok and not isinstance(value, bool)
                assert ok, (name, col, value)


def test_demand_reports_the_data_cutoff(client):
    body = client.get("/api/demand").json()
    as_of = date.fromisoformat(body["run"]["data_as_of"][:10])
    assert body["data_cutoff"] == {"date": (as_of - timedelta(days=6)).isoformat(), "excluded_days": 6}


def test_demand_without_the_metric_has_no_cutoff(client, fake_bigquery):
    rows = [r for r in data._fixture_table("model_metrics") if r["metric"] != "excluded_trailing_days"]
    fake_bigquery.tables = {"model_metrics": rows}
    assert client.get("/api/demand").json()["data_cutoff"] is None
