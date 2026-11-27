from typing import NamedTuple

from kfp import dsl

BASE_IMAGE = "python:3.12-slim"


@dsl.component(base_image=BASE_IMAGE)
def resolve_names(project: str, dataset: str, run_date: str) -> NamedTuple(
    "Names",
    run_date=str,
    churn_train=str,
    churn_score=str,
    clv_train=str,
    clv_score=str,
    demand_daily=str,
):
    import datetime
    from collections import namedtuple

    resolved = run_date or datetime.date.today().isoformat()
    datetime.date.fromisoformat(resolved)
    names = ("churn_train", "churn_score", "clv_train", "clv_score", "demand_daily")
    out = namedtuple("Names", ["run_date", *names])
    return out(resolved, *[f"`{project}.{dataset}.{n}`" for n in names])


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-bigquery==3.47.0"])
def data_checks(
    project: str,
    stats_query: str,
    run_date: str,
    churn_train: str,
    churn_score: str,
    clv_train: str,
    clv_score: str,
    demand_daily: str,
) -> NamedTuple(
    "Names",
    run_date=str,
    churn_train=str,
    churn_score=str,
    clv_train=str,
    clv_score=str,
    demand_daily=str,
):
    """Fail the run when a table is too small, has a feature newer than its cutoff, or an implausible label rate."""
    from google.cloud import bigquery

    min_rows = {
        "churn_train": 10000,
        "churn_score": 10000,
        "clv_train": 10000,
        "clv_score": 10000,
        "demand_daily": 5000,
    }
    label_bounds = {"churn_train": (0.5, 0.99), "clv_train": (0.02, 0.6)}
    min_series = 20

    client = bigquery.Client(project=project)
    job = client.query(stats_query, job_config=bigquery.QueryJobConfig(labels={"app": "vertex-demo"}))
    rows = [dict(r.items()) for r in job.result()]
    seen = {r["table_name"] for r in rows}
    problems = [f"{t}: no stats row" for t in min_rows if t not in seen]
    for r in rows:
        name = r["table_name"]
        if r["n_rows"] < min_rows.get(name, 1):
            problems.append(f"{name}: {r['n_rows']} rows, need at least {min_rows.get(name, 1)}")
        if r["max_source_ts"] is not None and r["cutoff"] is not None and r["max_source_ts"] >= r["cutoff"]:
            problems.append(f"{name}: source timestamp {r['max_source_ts']} is not before cutoff {r['cutoff']}")
        if name in label_bounds:
            low, high = label_bounds[name]
            rate = r["label_rate"]
            if rate is None or not low <= rate <= high:
                problems.append(f"{name}: label rate {rate} outside [{low}, {high}]")
        if name == "demand_daily" and (r["n_series"] or 0) < min_series:
            problems.append(f"{name}: {r['n_series']} series, need at least {min_series}")
    if problems:
        raise RuntimeError("data checks failed: " + "; ".join(problems))
    from collections import namedtuple

    out = namedtuple("Names", ["run_date", "churn_train", "churn_score", "clv_train", "clv_score", "demand_daily"])
    return out(run_date, churn_train, churn_score, clv_train, clv_score, demand_daily)
