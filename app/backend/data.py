import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

ALLOWED_DATASET = "vertex_demo_dashboard"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
CACHE_TTL = float(os.environ.get("DASHBOARD_CACHE_TTL", "60"))

TABLES = frozenset({
    "runs", "step_timings", "models", "model_metrics", "churn_curve_points", "churn_confusion",
    "churn_feature_importance", "churn_top_customers", "clv_predictions_sample", "clv_error_by_segment",
    "clv_deciles", "clv_tuning_trials", "demand_forecast", "demand_holdout_metrics",
})
JSON_COLUMNS = {"parameters", "hyperparameters"}


class DatasetError(ValueError):
    pass


def dataset() -> str:
    name = os.environ.get("BQ_DATASET", ALLOWED_DATASET)
    if name != ALLOWED_DATASET:
        raise DatasetError(f"dataset {name!r} is not allowed; only {ALLOWED_DATASET} can be read")
    return name


def fixture_mode() -> bool:
    return os.environ.get("DASHBOARD_FIXTURES", "").lower() in {"1", "true", "yes"}


def table(name: str) -> str:
    if name not in TABLES:
        raise DatasetError(f"table {name!r} is not part of the publish schema")
    return f"`{os.environ['BQ_PROJECT']}.{dataset()}.{name}`"


def query(sql: str, params: dict | None = None) -> list[dict]:
    from google.cloud import bigquery

    client = bigquery.Client(project=os.environ["BQ_PROJECT"])
    config = bigquery.QueryJobConfig(
        labels={"app": "vertex-demo"},
        query_parameters=[bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in (params or {}).items()],
    )
    try:
        return [dict(r) for r in client.query(sql, job_config=config).result()]
    except Exception as exc:  # the publish step creates the tables, so before the first publish they don't exist
        if type(exc).__name__ == "NotFound":
            return []
        raise


@cache
def _fixture_table(name: str) -> tuple[dict, ...]:
    return tuple(json.loads((FIXTURE_DIR / f"{name}.json").read_text()))


def _clean(row: dict) -> dict:
    for col in JSON_COLUMNS & row.keys():
        for _ in range(2):  # some columns arrive double-encoded (a JSON string holding a JSON string)
            if isinstance(row[col], str):
                row[col] = json.loads(row[col])
    return row


def _read(name: str, run_id: str | None) -> list[dict]:
    if fixture_mode():
        dataset()
        if name not in TABLES:
            raise DatasetError(f"table {name!r} is not part of the publish schema")
        rows = [dict(r) for r in _fixture_table(name) if run_id is None or r["run_id"] == run_id]
    elif run_id is None:
        rows = query(f"SELECT * FROM {table(name)}")
    else:
        rows = query(f"SELECT * FROM {table(name)} WHERE run_id = @run_id", {"run_id": run_id})
    return [_clean(r) for r in rows]


_cache: dict[tuple, tuple[float, list[dict]]] = {}
_refreshing: set[tuple] = set()
_lock = threading.Lock()


def _store(key: tuple, name: str, run_id: str | None) -> list[dict]:
    fresh = _read(name, run_id)
    with _lock:
        _cache[key] = (time.monotonic(), fresh)
        _refreshing.discard(key)
    return fresh


def _refresh_in_background(key: tuple, name: str, run_id: str | None) -> None:
    def work():
        try:
            _store(key, name, run_id)
        except Exception:  # keep serving the stale rows; the next request tries again
            with _lock:
                _refreshing.discard(key)

    threading.Thread(target=work, daemon=True).start()


def rows(name: str, run_id: str | None = None) -> list[dict]:
    """Rows of one publish-schema table, for one run or all runs.

    Cached for CACHE_TTL seconds; after that the stale rows are served while one background
    refresh runs, so a request never waits for BigQuery once the cache is warm."""
    key = (fixture_mode(), name, run_id)
    with _lock:
        hit = _cache.get(key)
        if hit:
            if time.monotonic() - hit[0] < CACHE_TTL:
                return hit[1]
            if key not in _refreshing:
                _refreshing.add(key)
                _refresh_in_background(key, name, run_id)
            return hit[1]
    return _store(key, name, run_id)


def prefetch(names: list[str], run_id: str | None = None) -> None:
    """Read several tables at once (BigQuery round trips overlap), so a cold page costs one round trip."""
    cold = [n for n in names if (fixture_mode(), n, run_id) not in _cache]
    if len(cold) > 1 and not fixture_mode():
        with ThreadPoolExecutor(max_workers=len(cold)) as pool:
            list(pool.map(lambda n: rows(n, run_id), cold))


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _refreshing.clear()
