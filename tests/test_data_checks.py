import datetime as dt

import pytest

from pipeline.components import data_checks

CUT = dt.datetime(2026, 8, 27)
GOOD = [
    {
        "table_name": "churn_train",
        "n_rows": 60000,
        "max_source_ts": CUT - dt.timedelta(seconds=1),
        "cutoff": CUT,
        "label_rate": 0.9,
        "n_series": None,
    },
    {
        "table_name": "churn_score",
        "n_rows": 70000,
        "max_source_ts": CUT - dt.timedelta(days=1),
        "cutoff": CUT,
        "label_rate": None,
        "n_series": None,
    },
    {
        "table_name": "clv_train",
        "n_rows": 50000,
        "max_source_ts": CUT - dt.timedelta(days=1),
        "cutoff": CUT,
        "label_rate": 0.1,
        "n_series": None,
    },
    {
        "table_name": "clv_score",
        "n_rows": 70000,
        "max_source_ts": CUT - dt.timedelta(days=1),
        "cutoff": CUT,
        "label_rate": None,
        "n_series": None,
    },
    {
        "table_name": "demand_daily",
        "n_rows": 18000,
        "max_source_ts": CUT - dt.timedelta(days=1),
        "cutoff": CUT,
        "label_rate": None,
        "n_series": 26,
    },
]


class FakeRow:
    def __init__(self, d):
        self.d = d

    def items(self):
        return self.d.items()


@pytest.fixture
def run(monkeypatch):
    from google.cloud import bigquery

    def go(rows):
        class Job:
            def result(self):
                return [FakeRow(r) for r in rows]

        class Client:
            def __init__(self, project):
                self.project = project

            def query(self, sql, job_config):
                assert job_config.labels == {"app": "vertex-demo"}
                return Job()

        monkeypatch.setattr(bigquery, "Client", Client)
        names = {k: k for k in ("churn_train", "churn_score", "clv_train", "clv_score", "demand_daily")}
        return data_checks.python_func("p", "select 1", "2026-11-25", **names)

    return go


def mutate(table, **changes):
    return [{**r, **changes} if r["table_name"] == table else r for r in GOOD]


def test_passes_and_forwards_names(run):
    out = run(GOOD)
    assert out.run_date == "2026-11-25" and out.churn_train == "churn_train"


def test_fails_when_a_feature_is_after_its_cutoff(run):
    with pytest.raises(RuntimeError, match="not before cutoff"):
        run(mutate("churn_train", max_source_ts=CUT + dt.timedelta(seconds=1)))


def test_fails_when_a_feature_is_exactly_at_the_cutoff(run):
    with pytest.raises(RuntimeError, match="not before cutoff"):
        run(mutate("clv_score", max_source_ts=CUT))


def test_fails_on_small_tables(run):
    with pytest.raises(RuntimeError, match="rows"):
        run(mutate("clv_train", n_rows=10))


@pytest.mark.parametrize("table,rate", [("churn_train", 0.2), ("churn_train", 1.0), ("clv_train", 0.0)])
def test_fails_on_implausible_label_rate(run, table, rate):
    with pytest.raises(RuntimeError, match="label rate"):
        run(mutate(table, label_rate=rate))


def test_fails_on_few_demand_series(run):
    with pytest.raises(RuntimeError, match="series"):
        run(mutate("demand_daily", n_series=3))


def test_fails_when_a_table_is_missing(run):
    with pytest.raises(RuntimeError, match="no stats row"):
        run(GOOD[:-1])
