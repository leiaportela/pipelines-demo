from datetime import date, timedelta

from fastapi import HTTPException

from app.backend import data

MODELS = ("churn", "clv", "demand")


def _time(row: dict, key: str):
    return str(row.get(key) or "")


def history() -> list[dict]:
    return sorted(data.rows("runs"), key=lambda r: _time(r, "started_at"), reverse=True)


class NoRun(Exception):
    """The dataset has no successful run yet (it is empty before the first publish)."""


EMPTY = {"empty": True, "run": None, "history": [], "message": "No pipeline run has been published yet."}


def resolve_run(run_id: str | None) -> dict:
    runs = history()
    if run_id:
        found = next((r for r in runs if r["run_id"] == run_id), None)
        if not found:
            raise HTTPException(404, f"run {run_id!r} not found")
        return found
    found = next((r for r in runs if r["status"] == "SUCCEEDED"), None)
    if not found:
        raise NoRun
    return found


def _by_model(run_id: str) -> dict[str, dict]:
    models = {m["model"]: m for m in data.rows("models", run_id)}
    metrics: dict[str, dict] = {m: {} for m in MODELS}
    for r in data.rows("model_metrics", run_id):
        metrics.setdefault(r["model"], {})[r["metric"]] = r["value"]
    return {m: {**models[m], "metrics": metrics.get(m, {})} for m in models}


def empty() -> dict:
    return {**EMPTY, "history": history()}


def overview(run_id: str | None) -> dict:
    run = resolve_run(run_id)
    data.prefetch(["step_timings", "models", "model_metrics"], run["run_id"])
    # The pipeline job's own root task is published as a step too; it is not a stage.
    steps = [s for s in data.rows("step_timings", run["run_id"])
             if not str(s["step"]).startswith(("vertex_demo", "vertex-demo"))]
    order = {"shared": 0, "churn": 1, "clv": 1, "demand": 1}
    steps.sort(key=lambda s: (_time(s, "started_at") or "~", order.get(s["branch"], 9)))
    return {"run": run, "steps": steps, "models": _by_model(run["run_id"]), "history": history()}


PAGE_TABLES = {
    "churn": ["churn_curve_points", "churn_confusion", "churn_feature_importance", "churn_top_customers"],
    "clv": ["clv_predictions_sample", "clv_error_by_segment", "clv_deciles", "clv_tuning_trials"],
    "demand": ["demand_forecast", "demand_holdout_metrics"],
}


def _model_page(run_id: str | None, model: str) -> tuple[dict, dict]:
    run = resolve_run(run_id)
    data.prefetch(["models", "model_metrics", *PAGE_TABLES[model]], run["run_id"])
    info = _by_model(run["run_id"]).get(model)
    return run, {"run": run, "model": info}


def churn(run_id: str | None) -> dict:
    run, out = _model_page(run_id, "churn")
    rid = run["run_id"]
    curves = {"pr": [], "roc": []}
    for r in sorted(data.rows("churn_curve_points", rid), key=lambda r: (r["x"], r["y"])):
        curves.setdefault(r["curve"], []).append(
            {"threshold": r["threshold"], "x": r["x"], "y": r["y"]})
    out["curves"] = curves
    out["confusion"] = sorted(data.rows("churn_confusion", rid), key=lambda r: r["threshold"])
    out["importance"] = sorted(data.rows("churn_feature_importance", rid), key=lambda r: r["rank"])
    out["customers"] = sorted(data.rows("churn_top_customers", rid), key=lambda r: r["rank"])
    return out


def clv(run_id: str | None) -> dict:
    run, out = _model_page(run_id, "clv")
    rid = run["run_id"]
    out["sample"] = data.rows("clv_predictions_sample", rid)
    out["segments"] = data.rows("clv_error_by_segment", rid)
    out["deciles"] = sorted(data.rows("clv_deciles", rid), key=lambda r: r["decile"])
    out["trials"] = sorted(data.rows("clv_tuning_trials", rid), key=lambda r: int(r["trial_id"]))
    return out


def _data_cutoff(run: dict, model: dict | None) -> dict | None:
    """The date the forecast was trained up to: data_as_of minus the trailing days the pipeline excluded.

    None when the run did not publish `excluded_trailing_days` (older runs)."""
    raw = ((model or {}).get("metrics") or {}).get("excluded_trailing_days")
    try:
        days = int(float(raw))
        as_of = date.fromisoformat(str(run.get("data_as_of"))[:10])
    except (TypeError, ValueError):
        return None
    if days < 0:
        return None
    return {"date": (as_of - timedelta(days=days)).isoformat(), "excluded_days": days}


def demand(run_id: str | None) -> dict:
    run, out = _model_page(run_id, "demand")
    rid = run["run_id"]
    series: dict[str, list] = {}
    for r in sorted(data.rows("demand_forecast", rid), key=lambda r: (r["category"], str(r["date"]))):
        series.setdefault(r["category"], []).append(
            [str(r["date"]), r["units"], r["forecast_units"], r["lower_bound"], r["upper_bound"]])
    out["columns"] = ["date", "units", "forecast_units", "lower_bound", "upper_bound"]
    out["series"] = series
    out["holdout"] = sorted(data.rows("demand_holdout_metrics", rid), key=lambda r: r["category"])
    out["data_cutoff"] = _data_cutoff(run, out.get("model"))
    return out
