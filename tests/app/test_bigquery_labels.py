"""Every BigQuery job the dashboard API starts must carry the app=vertex-demo label.

The BigQuery client is faked at the library boundary (google.cloud.bigquery), not at data.query,
so the real query() code builds the job config (the fake lives in conftest.py).
"""
from fastapi.testclient import TestClient

from app.backend.main import create_app

PAGES = ["/api/runs", "/api/overview", "/api/churn", "/api/clv", "/api/demand"]


def test_every_query_job_carries_the_app_label(fake_bigquery):
    client = TestClient(create_app())
    for path in PAGES:
        assert client.get(path).status_code == 200, path
    assert len(fake_bigquery.jobs) >= 10
    for sql, config in fake_bigquery.jobs:
        assert config.labels == {"app": "vertex-demo"}, sql
        assert "vertex_demo_dashboard" in sql
