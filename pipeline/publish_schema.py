"""BigQuery schemas of vertex_demo_dashboard: the contract between pipeline and dashboard.

Every table carries run_id. The latest successful run is the newest row in `runs` with status
SUCCEEDED; history is the whole table.
"""

import json
import sys
from pathlib import Path

DATASET = "vertex_demo_dashboard"
CONTRACT_PATH = Path(__file__).resolve().parent / "publish-schema.json"


def _f(name, type_, description, mode="NULLABLE"):
    return {"name": name, "type": type_, "mode": mode, "description": description}


def _run_id():
    return _f("run_id", "STRING", "Pipeline run id, key of every table", "REQUIRED")


TABLES: dict[str, dict] = {
    "runs": {
        "description": "One row per pipeline run",
        "fields": [
            _run_id(),
            _f("status", "STRING", "SUCCEEDED, FAILED or RUNNING", "REQUIRED"),
            _f("started_at", "TIMESTAMP", "Run start"),
            _f("ended_at", "TIMESTAMP", "Run end"),
            _f("parameters", "JSON", "Pipeline parameters as submitted"),
            _f("train_automl", "BOOL", "Whether AutoML trained in this run"),
            _f("data_as_of", "DATE", "Data cutoff, the run date"),
            _f("console_url", "STRING", "Vertex AI console link to the run"),
        ],
    },
    "step_timings": {
        "description": "Duration and status per pipeline step",
        "fields": [
            _run_id(),
            _f("step", "STRING", "Step name, for example build_features", "REQUIRED"),
            _f("branch", "STRING", "shared, churn, clv or demand"),
            _f("status", "STRING", "SUCCEEDED, FAILED or SKIPPED"),
            _f("started_at", "TIMESTAMP", "Step start"),
            _f("ended_at", "TIMESTAMP", "Step end"),
            _f("duration_seconds", "FLOAT64", "Step duration"),
        ],
    },
    "models": {
        "description": "One row per model per run, with how it was trained",
        "fields": [
            _run_id(),
            _f("model", "STRING", "churn, clv or demand", "REQUIRED"),
            _f(
                "training_method",
                "STRING",
                "AutoML Tabular, Custom XGBoost + tuning, BigQuery ML ARIMA_PLUS",
            ),
            _f("model_resource_name", "STRING", "Model Registry resource name or BigQuery model id"),
            _f("model_version", "STRING", "Model version"),
            _f("trained_at", "TIMESTAMP", "Training date"),
            _f("headline_metric_name", "STRING", "For example AUC-PR, RMSE, MAPE"),
            _f("headline_metric_value", "FLOAT64", "Headline metric value"),
            _f("console_url", "STRING", "Console link to the model or job"),
            _f("machine_type", "STRING", "Training machine"),
            _f("budget", "STRING", "Budget, for example 1000 milli node hours"),
            _f("trial_count", "INT64", "Tuning trials run"),
            _f("training_duration_seconds", "FLOAT64", "Training duration"),
            _f("trained_in_run", "BOOL", "False when an earlier model was reused"),
        ],
    },
    "model_metrics": {
        "description": "All scalar metrics per model, long format",
        "fields": [
            _run_id(),
            _f("model", "STRING", "churn, clv or demand", "REQUIRED"),
            _f("metric", "STRING", "Metric name", "REQUIRED"),
            _f("value", "FLOAT64", "Metric value"),
        ],
    },
    "churn_curve_points": {
        "description": "Precision-recall and ROC curve points",
        "fields": [
            _run_id(),
            _f("curve", "STRING", "pr or roc", "REQUIRED"),
            _f("threshold", "FLOAT64", "Score threshold"),
            _f("x", "FLOAT64", "Recall (pr) or false positive rate (roc)"),
            _f("y", "FLOAT64", "Precision (pr) or true positive rate (roc)"),
        ],
    },
    "churn_confusion": {
        "description": "Confusion matrix counts by threshold",
        "fields": [
            _run_id(),
            _f("threshold", "FLOAT64", "Score threshold", "REQUIRED"),
            _f("tp", "INT64", "True positives"),
            _f("fp", "INT64", "False positives"),
            _f("tn", "INT64", "True negatives"),
            _f("fn", "INT64", "False negatives"),
        ],
    },
    "churn_feature_importance": {
        "description": "Global feature importance",
        "fields": [
            _run_id(),
            _f("feature", "STRING", "Feature name", "REQUIRED"),
            _f("importance", "FLOAT64", "Importance"),
            _f("rank", "INT64", "1 is most important"),
        ],
    },
    "churn_top_customers": {
        "description": "Top at-risk customers with top 3 attributions",
        "fields": [
            _run_id(),
            _f("user_id", "INT64", "Customer id", "REQUIRED"),
            _f("churn_score", "FLOAT64", "Predicted churn probability"),
            _f("rank", "INT64", "1 is most at risk"),
            _f("reason_1_feature", "STRING", "Top attribution feature"),
            _f("reason_1_value", "FLOAT64", "Top attribution value"),
            _f("reason_2_feature", "STRING", "Second attribution feature"),
            _f("reason_2_value", "FLOAT64", "Second attribution value"),
            _f("reason_3_feature", "STRING", "Third attribution feature"),
            _f("reason_3_value", "FLOAT64", "Third attribution value"),
        ],
    },
    "clv_predictions_sample": {
        "description": "Predicted vs actual spend sample",
        "fields": [
            _run_id(),
            _f("user_id", "INT64", "Customer id", "REQUIRED"),
            _f("predicted", "FLOAT64", "Predicted 180-day spend"),
            _f("actual", "FLOAT64", "Actual 180-day spend"),
            _f("segment", "STRING", "Customer segment"),
        ],
    },
    "clv_error_by_segment": {
        "description": "Error by customer segment",
        "fields": [
            _run_id(),
            _f("segment", "STRING", "Customer segment", "REQUIRED"),
            _f("customers", "INT64", "Customers in segment"),
            _f("mae", "FLOAT64", "Mean absolute error"),
            _f("rmse", "FLOAT64", "Root mean squared error"),
            _f("bias", "FLOAT64", "Mean predicted minus actual"),
        ],
    },
    "clv_deciles": {
        "description": "Revenue and lift by predicted decile",
        "fields": [
            _run_id(),
            _f("decile", "INT64", "1 is the highest predicted spend", "REQUIRED"),
            _f("customers", "INT64", "Customers in decile"),
            _f("predicted_revenue", "FLOAT64", "Sum of predicted spend"),
            _f("actual_revenue", "FLOAT64", "Sum of actual spend"),
            _f("lift", "FLOAT64", "Decile mean actual over overall mean actual"),
        ],
    },
    "clv_tuning_trials": {
        "description": "Every hyperparameter tuning trial",
        "fields": [
            _run_id(),
            _f("trial_id", "STRING", "Trial id", "REQUIRED"),
            _f("hyperparameters", "JSON", "Hyperparameters of the trial"),
            _f("rmse", "FLOAT64", "Validation RMSE"),
            _f("is_best", "BOOL", "Best trial"),
        ],
    },
    "demand_forecast": {
        "description": "Actuals and forecast with the 90% band per category per day",
        "fields": [
            _run_id(),
            _f("category", "STRING", "Product category", "REQUIRED"),
            _f("date", "DATE", "Day", "REQUIRED"),
            _f("kind", "STRING", "actual, holdout or forecast", "REQUIRED"),
            _f("units", "FLOAT64", "Actual units (actual, holdout)"),
            _f("forecast_units", "FLOAT64", "Forecast units (holdout, forecast)"),
            _f("lower_bound", "FLOAT64", "90% interval lower bound"),
            _f("upper_bound", "FLOAT64", "90% interval upper bound"),
        ],
    },
    "demand_holdout_metrics": {
        "description": "Holdout MAPE and interval coverage per category",
        "fields": [
            _run_id(),
            _f("category", "STRING", "Product category", "REQUIRED"),
            _f("mape", "FLOAT64", "Holdout MAPE"),
            _f("coverage", "FLOAT64", "Share of holdout days inside the 90% band"),
            _f("holdout_days", "INT64", "Holdout length in days"),
        ],
    },
}


def as_contract() -> dict:
    return {"dataset": DATASET, "tables": TABLES}


def render() -> str:
    return json.dumps(as_contract(), indent=2) + "\n"


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if CONTRACT_PATH.read_text() == render() else 1)
    CONTRACT_PATH.write_text(render())
    print(CONTRACT_PATH)
