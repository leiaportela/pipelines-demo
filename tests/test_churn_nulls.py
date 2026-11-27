import re
from types import SimpleNamespace

import pytest
from google.cloud import bigquery

from pipeline import churn
from pipeline import sql_builders as sb

VALUES = {"churn_automl_train": "`p.d.t`", "churn_automl_score": "`p.d.s`"}


def column_exprs(sql):
    body = sql.split("SELECT", 1)[1].split("FROM", 1)[0]
    return re.findall(r"COALESCE\((\w+), ([^)]+)\) AS (\w+)", body)


@pytest.mark.parametrize("builder", [sb.churn_automl_train, sb.churn_automl_score])
def test_every_feature_column_is_coalesced(builder):
    sql = builder()
    for column in sb.CHURN_FEATURES:
        assert re.search(rf"COALESCE\({column}, [^)]+\) AS {column}\b", sql), column


def test_days_since_last_event_defaults_past_the_lookback_window():
    assert sb.CHURN_NULL_DEFAULTS["days_since_last_event"] == sb.EVENT_LOOKBACK_DAYS + 1
    assert f"INTERVAL {sb.EVENT_LOOKBACK_DAYS} DAY" in sb._features("DATE('2026-01-01')")


def test_check_sql_covers_every_column_and_the_target():
    train = sb.automl_train_check()
    for column in sb.CHURN_FEATURES:
        assert f"AS invalid__{column}" in train
    for column in sb.CHURN_NUMERIC:
        assert f"IS_NAN(CAST({column} AS FLOAT64))" in train
        assert f"IS_INF(CAST({column} AS FLOAT64))" in train
    assert "COUNTIF(outcome IS NULL) AS null_target" in train
    assert "null_target" not in sb.automl_score_check()
    assert sb.render(train, **VALUES).endswith("FROM `p.d.t`")


class FakeRow:
    def __init__(self, data):
        self.data = data

    def items(self):
        return self.data.items()


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def __call__(self, project):
        return self

    def query(self, query, job_config=None):
        self.queries.append((query, job_config))
        row = self.rows["train" if "train" in query else "score"]
        return SimpleNamespace(result=lambda: [FakeRow(row)])


def clean(n=1000, target=True):
    row = {"n_rows": n, **{f"invalid__{c}": 0 for c in sb.CHURN_FEATURES}}
    if target:
        row["null_target"] = 0
    return row


def run_check(monkeypatch, train, score):
    client = FakeClient({"train": train, "score": score})
    monkeypatch.setattr(bigquery, "Client", client)
    return client, churn.churn_data_check.python_func(
        project="p", train_query="train", score_query="score", max_invalid_fraction=sb.MAX_INVALID_FRACTION
    )


def test_clean_tables_pass_with_labelled_queries(monkeypatch):
    client, result = run_check(monkeypatch, clean(), clean(target=False))
    assert result == "ok"
    assert all(cfg.labels == {"app": "vertex-demo"} for _, cfg in client.queries)


def test_null_in_a_numeric_column_fails_the_run(monkeypatch):
    bad = clean()
    bad["invalid__days_since_last_event"] = 900
    with pytest.raises(RuntimeError, match=r"churn_automl_train\.days_since_last_event: 900 of 1000"):
        run_check(monkeypatch, bad, clean(target=False))


def test_invalid_score_column_fails_the_run(monkeypatch):
    bad = clean(target=False)
    bad["invalid__age"] = 100
    with pytest.raises(RuntimeError, match=r"churn_automl_score\.age"):
        run_check(monkeypatch, clean(), bad)


def test_null_target_fails_the_run(monkeypatch):
    bad = clean()
    bad["null_target"] = 1
    with pytest.raises(RuntimeError, match="NULL target"):
        run_check(monkeypatch, bad, clean(target=False))


def test_small_invalid_fraction_is_tolerated_and_empty_tables_fail(monkeypatch):
    ok = clean()
    ok["invalid__age"] = 5
    assert run_check(monkeypatch, ok, clean(target=False))[1] == "ok"
    with pytest.raises(RuntimeError, match="no rows"):
        run_check(monkeypatch, clean(n=0), clean(target=False))
