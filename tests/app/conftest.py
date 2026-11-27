import re
import sys
import types

import pytest
from fastapi.testclient import TestClient

from app.backend import data
from app.backend.main import create_app


@pytest.fixture(autouse=True)
def fresh_cache():
    data.clear_cache()
    yield
    data.clear_cache()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_FIXTURES", "1")
    monkeypatch.setenv("BQ_PROJECT", "p")
    monkeypatch.delenv("BQ_DATASET", raising=False)
    return TestClient(create_app())


class NotFound(Exception):
    pass


class FakeBigQuery(types.ModuleType):
    def __init__(self):
        super().__init__("google.cloud.bigquery")
        self.jobs = []
        self.tables = {}  # name -> rows replacing the fixture rows
        self.missing = set()  # tables that do not exist yet (BigQuery answers NotFound)
        outer = self

        class QueryJobConfig:
            def __init__(self, labels=None, query_parameters=None, **kwargs):
                self.labels = labels
                self.query_parameters = query_parameters

        class ScalarQueryParameter:
            def __init__(self, name, type_, value):
                self.name, self.type_, self.value = name, type_, value

        class Job:
            def __init__(self, rows):
                self._rows = rows

            def result(self):
                return self._rows

        class Client:
            def __init__(self, project=None, **kwargs):
                self.project = project

            def query(self, sql, job_config=None, **kwargs):
                outer.jobs.append((sql, job_config))
                name = re.search(r"\.(\w+)`", sql).group(1)
                if name in outer.missing:
                    raise NotFound(f"Not found: Table {name}")
                rows = outer.tables.get(name, data._fixture_table(name))
                wanted = job_config.query_parameters[0].value if "@run_id" in sql else None
                return Job([dict(r) for r in rows if wanted is None or r["run_id"] == wanted])

        self.QueryJobConfig, self.ScalarQueryParameter, self.Client = QueryJobConfig, ScalarQueryParameter, Client


@pytest.fixture
def fake_bigquery(monkeypatch):
    fake = FakeBigQuery()
    cloud = types.ModuleType("google.cloud")
    cloud.bigquery = fake
    google = types.ModuleType("google")
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake)
    monkeypatch.delenv("DASHBOARD_FIXTURES", raising=False)
    monkeypatch.delenv("BQ_DATASET", raising=False)
    monkeypatch.setenv("BQ_PROJECT", "p")
    return fake
