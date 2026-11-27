import json
import re

import pytest
from kfp import compiler, dsl

from pipeline import forecast as fc

LABEL = {"app": "vertex-demo"}
VALUES = {
    "run_date": "2026-11-25",
    "demand_daily": "`p.d.demand_daily`",
    **{t: f"`p.d.{t}`" for t in fc.TABLES},
}


def sql(builder):
    return fc.sql_input(builder(), **VALUES)


END = fc.sql_input(fc._end_excl(), **VALUES)


def test_the_cutoff_detector_compares_recent_daily_totals_with_an_earlier_median():
    assert "IFNULL(MIN(sale_date), DATE('2026-11-25'))" in END
    assert "sale_date >= DATE_SUB(DATE('2026-11-25'), INTERVAL 14 DAY)" in END
    assert "total > 1.5 * (SELECT APPROX_QUANTILES(total, 2)[OFFSET(1)]" in END
    assert "sale_date >= DATE_SUB(DATE('2026-11-25'), INTERVAL 60 DAY)" in END
    assert "sale_date < DATE_SUB(DATE('2026-11-25'), INTERVAL 14 DAY)" in END
    assert "`p.d.demand_daily`" in END and "{" not in END


def test_holdout_model_trains_before_the_cutoff_minus_28_days_per_category():
    q = sql(fc.holdout_model_sql)
    assert "model_type = 'ARIMA_PLUS'" in q
    assert "time_series_id_col = 'category'" in q
    assert f"sale_date < DATE_SUB({END}, INTERVAL 28 DAY)" in q
    assert "`p.d.demand_holdout_model`" in q and "`p.d.demand_daily`" in q


def test_production_model_uses_the_series_up_to_the_cutoff_not_the_run_date():
    q = sql(fc.production_model_sql)
    assert f"sale_date < {END}" in q
    assert "sale_date < DATE('2026-11-25')" not in q
    assert "INTERVAL 28 DAY" not in q
    assert "time_series_id_col = 'category'" in q and "`p.d.demand_model`" in q


def test_holdout_forecast_scores_the_28_days_before_the_cutoff():
    q = sql(fc.holdout_forecast_sql)
    assert "STRUCT(28 AS horizon, 0.9 AS confidence_level)" in q
    assert f"a.sale_date >= DATE_SUB({END}, INTERVAL 28 DAY)" in q
    assert f"a.sale_date < {END}" in q
    assert "a.sale_date < DATE('2026-11-25')" not in q


def test_production_forecast_crosses_the_gap_and_keeps_30_days_from_the_run_date():
    q = sql(fc.forecast_sql)
    assert "STRUCT(44 AS horizon, 0.9 AS confidence_level)" in q
    assert "DATE(forecast_timestamp) >= DATE('2026-11-25')" in q
    assert "DATE(forecast_timestamp) < DATE_ADD(DATE('2026-11-25'), INTERVAL 30 DAY)" in q
    assert "`p.d.demand_model`" in q and "`p.d.demand_forecast`" in q


def test_the_excluded_days_are_reported_for_the_dashboard():
    q = sql(fc.excluded_days_sql)
    assert q.startswith("SELECT DATE_DIFF(DATE('2026-11-25'), (SELECT IFNULL(MIN(sale_date)")
    assert q.endswith("AS excluded_days")


@pytest.mark.parametrize(
    "builder,model",
    [(fc.label_holdout_model_sql, "demand_holdout_model"), (fc.label_production_model_sql, "demand_model")],
)
def test_both_models_get_the_app_label(builder, model):
    q = sql(builder)
    assert q == f"ALTER MODEL `p.d.{model}` SET OPTIONS (labels = [('app', 'vertex-demo')])"


def test_metrics_report_mape_coverage_per_category_and_overall():
    q = sql(fc.holdout_metrics_sql)
    assert "units BETWEEN lower_bound AND upper_bound" in q
    assert "IF(units > 0, ABS(units - forecast_units) / units, NULL)" in q
    assert "GROUP BY category" in q and "'ALL'" in q


@pytest.mark.parametrize("builder", [fc.holdout_model_sql, fc.production_model_sql, fc.holdout_forecast_sql])
def test_nothing_reads_data_on_or_after_the_run_date(builder):
    q = sql(builder)
    assert re.search(r"sale_date < (DATE_SUB\()?DATE\('2026-11-25'\)", q)


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    @dsl.pipeline(name="demand-only")
    def demand_only():
        fc.demand_branch(project="p", bq_dataset="d", run_date="2026-11-25", demand_daily="`p.d.demand_daily`")

    path = tmp_path_factory.mktemp("c") / "demand.json"
    compiler.Compiler().compile(demand_only, str(path))
    return json.loads(path.read_text())


def tasks_of(spec, prefix):
    return [
        t
        for comp in spec["components"].values()
        for t in comp.get("dag", {}).get("tasks", {}).values()
        if t["componentRef"]["name"].startswith(prefix)
    ]


def test_models_come_from_the_bigquery_ml_components(spec):
    assert len(tasks_of(spec, "comp-bigquery-create-model-job")) == 2
    assert len(tasks_of(spec, "comp-bigquery-query-job")) == 5


def test_every_bigquery_job_carries_the_app_label(spec):
    jobs = tasks_of(spec, "comp-bigquery-create-model-job") + tasks_of(spec, "comp-bigquery-query-job")
    assert jobs
    for task in jobs:
        assert task["inputs"]["parameters"]["labels"]["runtimeValue"]["constant"] == LABEL


def test_no_endpoint_or_deploy_component_and_nothing_cached(spec):
    names = " ".join(spec["components"]).lower()
    assert "endpoint" not in names and "deploy" not in names
    for comp in spec["components"].values():
        for task in comp.get("dag", {}).get("tasks", {}).values():
            assert task.get("cachingOptions", {}).get("enableCache") is not True
            assert "cachingOptions" in task


def test_holdout_is_evaluated_before_the_production_model_is_trained(spec):
    dag = spec["components"]["comp-demand-bqml-arima"]["dag"]["tasks"]
    key = {t["taskInfo"]["name"]: k for k, t in dag.items()}
    depends = {t["taskInfo"]["name"]: set(t.get("dependentTasks", [])) for t in dag.values()}
    assert key["evaluate_holdout"] in depends["train_production_arima_plus"]
    assert key["forecast_holdout"] in depends["evaluate_holdout"]
    assert key["label_holdout_model"] in depends["forecast_holdout"]
    assert key["train_holdout_arima_plus"] in depends["label_holdout_model"]
    assert key["label_production_model"] in depends["forecast_30_days"]
    assert key["train_production_arima_plus"] in depends["label_production_model"]
    assert key["forecast_30_days"] in depends["check_forecast"]


def test_component_sources_run_standalone_in_the_container(spec):
    """The container only has the function source: a module-level name in a signature fails there."""
    executors = spec["deploymentSpec"]["executors"]
    assert executors
    for name, ex in executors.items():
        source = ex["container"]["command"][-1]
        if "def " not in source:
            continue
        exec(compile(source, name, "exec"), {})


def test_the_full_pipeline_runs_the_demand_branch_without_train_automl(tmp_path):
    from pipeline.compile import compile_pipeline

    full = json.loads(compile_pipeline(tmp_path / "pipeline.json").read_text())
    root = full["components"]["comp-exit-handler-1"]["dag"]["tasks"]
    demand = [t for t in root.values() if t["componentRef"]["name"] == "comp-demand-bqml-arima"]
    assert len(demand) == 1
    assert "triggerPolicy" not in demand[0]
    assert "train_automl" not in json.dumps(demand[0]["inputs"])
