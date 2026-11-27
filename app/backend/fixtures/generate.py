"""Generates the fixture rows for every publish-schema table.

Run: uv run python app/backend/fixtures/generate.py
"""

import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent
CONSOLE = "https://console.cloud.google.com/vertex-ai"
PROJECT = "your-project-id"

CATEGORIES = [
    "Accessories", "Active", "Blazers & Jackets", "Clothing Sets", "Dresses", "Fashion Hoodies & Sweatshirts",
    "Intimates", "Jeans", "Jumpsuits & Rompers", "Leggings", "Maternity", "Outerwear & Coats", "Pants",
    "Pants & Capris", "Plus", "Shorts", "Skirts", "Sleep & Lounge", "Socks", "Socks & Hosiery", "Suits",
    "Suits & Sport Coats", "Sweaters", "Swim", "Tops & Tees", "Underwear",
]
SEGMENTS = ["One-time", "Occasional", "Regular", "High value"]
FEATURES = [
    "days_since_last_order", "orders_90d", "orders_total", "avg_order_value", "sessions_30d", "return_rate",
    "days_since_signup", "items_per_order", "discount_share", "email_opt_in", "traffic_source", "distinct_categories",
]
RUNS = [
    # id, start, status, train_automl, rows for the detail tables
    ("run-20261104-0600", datetime(2026, 11, 4, 6, 0, tzinfo=UTC), "SUCCEEDED", True, False),
    ("run-20261111-0600", datetime(2026, 11, 11, 6, 0, tzinfo=UTC), "SUCCEEDED", False, True),
    ("run-20261118-0600", datetime(2026, 11, 18, 6, 0, tzinfo=UTC), "FAILED", False, False),
    ("run-20261125-0600", datetime(2026, 11, 25, 6, 0, tzinfo=UTC), "SUCCEEDED", False, True),
]
STEPS = [
    ("build_features", "shared", 210), ("check_data", "shared", 45),
    ("train_automl_churn", "churn", 4980), ("batch_predict_churn", "churn", 420),
    ("train_clv_tuning", "clv", 1860), ("batch_predict_clv", "clv", 390),
    ("train_demand_arima", "demand", 330), ("forecast_demand", "demand", 95),
    ("publish", "shared", 60),
]
P_POS = 0.07
N = 18240


def tpr(t):
    return 1 - t**2.2


def fpr(t):
    return 1 - t**0.55


def precision(t):
    a, b = P_POS * tpr(t), (1 - P_POS) * fpr(t)
    return a / (a + b) if a + b else 1.0


def iso(d):
    return d.isoformat().replace("+00:00", "Z")


def run_header(tables, rid, start, status, automl):
    rnd = random.Random(rid)
    day = start.date()
    failed = status == "FAILED"
    total = 0.0
    t = start
    for step, branch, secs in STEPS:
        secs *= rnd.uniform(0.92, 1.08)
        st = "SUCCEEDED"
        if step == "train_automl_churn" and not automl:
            st = "SKIPPED"
        if failed and step == "train_clv_tuning":
            st = "FAILED"
        elif failed and step in ("batch_predict_clv", "publish", "train_demand_arima", "forecast_demand"):
            st = "SKIPPED"
        dur = 0.0 if st == "SKIPPED" else secs
        total += dur
        tables["step_timings"].append({
            "run_id": rid, "step": step, "branch": branch, "status": st,
            "started_at": iso(t) if dur else None, "ended_at": iso(t + timedelta(seconds=dur)) if dur else None,
            "duration_seconds": round(dur, 1)})
        t += timedelta(seconds=dur)
    tables["runs"].append({
        "run_id": rid, "status": status, "started_at": iso(start),
        "ended_at": iso(start + timedelta(seconds=total + 120)),
        "parameters": {"train_automl": automl, "tuning_trials": 12, "forecast_horizon_days": 30},
        "train_automl": automl, "data_as_of": day.isoformat(),
        "console_url": f"{CONSOLE}/locations/us-central1/pipelines/runs/{rid}?project={PROJECT}"})


def run_models(tables, rid, start, automl, idx):
    drift = idx * 0.004
    metrics = {
        "churn": {"auc_pr": 0.227 + drift, "auc_roc": 0.761 + drift, "log_loss": 0.47 - drift,
                  "positive_rate": P_POS, "test_customers": N},
        "clv": {"rmse": 44.0 - idx * 0.1, "mae": 24.6 - idx * 0.2, "r2": 0.033 + drift, "baseline_rmse": 44.8,
                "test_customers": N},
        "demand": {"mape_mean": 14.2 - idx * 0.3, "coverage_mean": 0.88 + drift, "categories": 26,
                   "excluded_trailing_days": 6},
    }
    for model, ms in metrics.items():
        for k, v in ms.items():
            tables["model_metrics"].append({"run_id": rid, "model": model, "metric": k, "value": round(float(v), 4)})
    trained = start + timedelta(hours=2)
    tables["models"] += [
        {"run_id": rid, "model": "churn", "training_method": "AutoML Tabular",
         "model_resource_name": f"projects/{PROJECT}/locations/us-central1/models/41720{idx}",
         "model_version": "1" if idx == 0 else "2",
         "trained_at": iso(trained if automl else datetime(2026, 11, 4, 8, 0, tzinfo=UTC)),
         "headline_metric_name": "AUC-PR", "headline_metric_value": round(metrics["churn"]["auc_pr"], 4),
         "console_url": f"{CONSOLE}/models/41720{idx}?project={PROJECT}", "machine_type": "AutoML managed",
         "budget": "1,000 milli node hours", "trial_count": None,
         "training_duration_seconds": 4980.0 if automl else None, "trained_in_run": automl},
        {"run_id": rid, "model": "clv", "training_method": "Custom XGBoost + hyperparameter tuning",
         "model_resource_name": f"projects/{PROJECT}/locations/us-central1/models/55310{idx}",
         "model_version": str(idx + 1), "trained_at": iso(trained),
         "headline_metric_name": "RMSE", "headline_metric_value": round(metrics["clv"]["rmse"], 4),
         "console_url": f"{CONSOLE}/models/55310{idx}?project={PROJECT}", "machine_type": "n1-standard-4 (CPU)",
         "budget": "12 trials, 3 in parallel", "trial_count": 12, "training_duration_seconds": 1860.0,
         "trained_in_run": True},
        {"run_id": rid, "model": "demand", "training_method": "BigQuery ML ARIMA_PLUS",
         "model_resource_name": f"{PROJECT}.vertex_demo_features.demand_arima_plus",
         "model_version": str(idx + 1), "trained_at": iso(trained),
         "headline_metric_name": "MAPE", "headline_metric_value": round(metrics["demand"]["mape_mean"], 4),
         "console_url": f"https://console.cloud.google.com/bigquery?project={PROJECT}",
         "machine_type": "BigQuery on-demand", "budget": "One time series per category", "trial_count": None,
         "training_duration_seconds": 330.0, "trained_in_run": True},
    ]
    return metrics


def churn_rows(tables, rid, rnd):
    ts = [i / 50 for i in range(1, 50)]
    for th in ts:
        tables["churn_curve_points"].append({"run_id": rid, "curve": "pr", "threshold": round(th, 3),
                                             "x": round(tpr(th), 4), "y": round(precision(th), 4)})
    for th in [0.0, *ts, 1.0]:
        tables["churn_curve_points"].append({"run_id": rid, "curve": "roc", "threshold": round(th, 3),
                                             "x": round(fpr(th), 4), "y": round(tpr(th), 4)})
    for k in range(1, 20):
        th = k / 20
        tp = round(N * P_POS * tpr(th))
        fp = round(N * (1 - P_POS) * fpr(th))
        tables["churn_confusion"].append({
            "run_id": rid, "threshold": th, "tp": tp, "fp": fp,
            "tn": round(N * (1 - P_POS)) - fp, "fn": round(N * P_POS) - tp})
    weights = [0.21, 0.16, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03]
    for rank, (f, w) in enumerate(zip(FEATURES, weights, strict=True), 1):
        tables["churn_feature_importance"].append({"run_id": rid, "feature": f, "importance": w, "rank": rank})
    for rank in range(1, 16):
        row = {"run_id": rid, "user_id": rnd.randint(10000, 99999),
               "churn_score": round(0.97 - rank * 0.012 + rnd.uniform(-0.004, 0.004), 4), "rank": rank}
        for i, f in enumerate(rnd.sample(FEATURES[:8], 3), 1):
            row[f"reason_{i}_feature"] = f
            row[f"reason_{i}_value"] = round(rnd.uniform(0.04, 0.3) * (1 if rnd.random() < 0.8 else -1) / i, 4)
        tables["churn_top_customers"].append(row)


def clv_rows(tables, rid, rnd, rmse):
    shares = [0.46, 0.30, 0.17, 0.07]
    means = [28, 74, 168, 412]
    sample = []
    for _ in range(420):
        s = rnd.choices(range(4), shares)[0]
        pred = max(2.0, rnd.gauss(means[s], means[s] * 0.28))
        actual = max(0.0, pred * math.exp(rnd.gauss(-0.02, 0.32 - s * 0.03)) + rnd.gauss(0, 4))
        sample.append((s, round(pred, 2), round(actual, 2)))
        tables["clv_predictions_sample"].append({
            "run_id": rid, "user_id": rnd.randint(10000, 99999), "predicted": round(pred, 2),
            "actual": round(actual, 2), "segment": SEGMENTS[s]})
    for s, name in enumerate(SEGMENTS):
        errs = [p - a for si, p, a in sample if si == s]
        tables["clv_error_by_segment"].append({
            "run_id": rid, "segment": name, "customers": round(N * shares[s]),
            "mae": round(sum(abs(e) for e in errs) / len(errs), 2),
            "rmse": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 2),
            "bias": round(sum(errs) / len(errs), 2)})
    order = sorted(sample, key=lambda r: r[1])
    mean_actual = sum(r[2] for r in sample) / len(sample)
    size = len(order) // 10
    for d in range(10):
        chunk = order[d * size:(d + 1) * size]
        cust = N // 10
        a = sum(r[2] for r in chunk) / len(chunk)
        tables["clv_deciles"].append({
            "run_id": rid, "decile": 10 - d, "customers": cust,
            "predicted_revenue": round(sum(r[1] for r in chunk) / len(chunk) * cust, 2),
            "actual_revenue": round(a * cust, 2), "lift": round(a / mean_actual, 3)})
    rmses = [round(rnd.uniform(rmse + 2.5, rmse + 20) - i * 0.4, 3) for i in range(12)]
    best = rnd.randint(5, 11)
    rmses[best] = rmse
    for i, r in enumerate(rmses):
        tables["clv_tuning_trials"].append({
            "run_id": rid, "trial_id": str(i + 1),
            "hyperparameters": {"max_depth": rnd.randint(3, 9), "learning_rate": round(rnd.uniform(0.02, 0.3), 3),
                                "subsample": round(rnd.uniform(0.6, 1.0), 2),
                                "n_estimators": rnd.choice([200, 400, 600])},
            "rmse": r, "is_best": i == best})


def demand_rows(tables, rid, rnd, day):
    for c, cat in enumerate(CATEGORIES):
        base = 20 + (c * 37) % 90
        tables["demand_holdout_metrics"].append({
            "run_id": rid, "category": cat, "mape": round(rnd.uniform(8, 22), 2),
            "coverage": round(rnd.uniform(0.82, 0.95), 3), "holdout_days": 28})
        for k in range(-83, 31):
            d = day + timedelta(days=k)
            level = base * (1 + 0.003 * k) * (1 + 0.18 * math.sin(2 * math.pi * (d.weekday() - 4) / 7))
            row = {"run_id": rid, "category": cat, "date": d.isoformat(), "kind": "actual", "units": None,
                   "forecast_units": None, "lower_bound": None, "upper_bound": None}
            if k <= 0:
                row["units"] = round(max(0.0, level + rnd.gauss(0, base * 0.09)), 1)
            else:
                w = base * (0.14 + 0.012 * k)
                row.update(kind="forecast", forecast_units=round(level, 1),
                           lower_bound=round(max(0.0, level - w), 1), upper_bound=round(level + w, 1))
            tables["demand_forecast"].append(row)


def main():
    names = ["runs", "step_timings", "models", "model_metrics", "churn_curve_points", "churn_confusion",
             "churn_feature_importance", "churn_top_customers", "clv_predictions_sample", "clv_error_by_segment",
             "clv_deciles", "clv_tuning_trials", "demand_forecast", "demand_holdout_metrics"]
    tables = {n: [] for n in names}
    ok = 0
    for rid, start, status, automl, detail in RUNS:
        run_header(tables, rid, start, status, automl)
        if status == "SUCCEEDED":
            metrics = run_models(tables, rid, start, automl, ok)
            if detail:
                rnd = random.Random(rid + "detail")
                churn_rows(tables, rid, rnd)
                clv_rows(tables, rid, rnd, metrics["clv"]["rmse"])
                demand_rows(tables, rid, rnd, start.date())
            ok += 1
    for name, rows in tables.items():
        (OUT / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
