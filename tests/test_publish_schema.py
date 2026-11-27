import json

from pipeline import publish_schema as ps


def test_contract_json_matches_code():
    assert json.loads(ps.CONTRACT_PATH.read_text()) == ps.as_contract()


def test_every_table_keyed_by_run_id():
    for name, table in ps.TABLES.items():
        first = table["fields"][0]
        assert first["name"] == "run_id" and first["mode"] == "REQUIRED", name


def test_covers_dashboard_pages():
    expected = {
        "runs",
        "step_timings",
        "models",
        "model_metrics",
        "churn_curve_points",
        "churn_confusion",
        "churn_feature_importance",
        "churn_top_customers",
        "clv_predictions_sample",
        "clv_error_by_segment",
        "clv_deciles",
        "clv_tuning_trials",
        "demand_forecast",
        "demand_holdout_metrics",
    }
    assert expected <= set(ps.TABLES)


def test_tuning_trials_hold_best_flag_and_rmse():
    names = {f["name"] for f in ps.TABLES["clv_tuning_trials"]["fields"]}
    assert {"trial_id", "hyperparameters", "rmse", "is_best"} <= names
