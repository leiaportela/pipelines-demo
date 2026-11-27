import re
from typing import NamedTuple

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from kfp import dsl

from pipeline import sql_builders as sb
from pipeline.churn import TARGET_COLUMN, churn_branch
from pipeline.clv import clv_branch
from pipeline.components import data_checks, resolve_names
from pipeline.forecast import demand_branch
from pipeline.publish import (
    CLV_SAMPLE_ROWS,
    DEMAND_ACTUAL_DAYS,
    TOP_CUSTOMERS,
    publish_results,
    record_run,
    schema_json,
)

LABELS = {"app": "vertex-demo"}
BQ_LOCATION = "US"


def sql_input(template: str, **values):
    """Turn a SQL template into a string input; token values may be task outputs resolved at run time."""
    pieces = [p for p in re.split(r"(\{\w+\})", template) if p]
    return "".join(f"{values[p[1:-1]]}" if p[1:-1] in values else p for p in pieces)


@dsl.pipeline(name="build-features")
def build_features(project: str, location: str, run_date: str, bq_dataset: str) -> NamedTuple(
    "Features",
    run_date=str,
    churn_train=str,
    churn_score=str,
    clv_train=str,
    clv_score=str,
    demand_daily=str,
):
    names = resolve_names(project=project, dataset=bq_dataset, run_date=run_date).set_caching_options(False)
    values = {"run_date": names.outputs["run_date"], **{t: names.outputs[t] for t in sb.TABLES}}
    builds = [
        BigqueryQueryJobOp(
            query=sql_input(builder(), **values),
            project=project,
            location=BQ_LOCATION,
            labels=LABELS,
        )
        .set_display_name(f"build_{name}")
        .set_caching_options(False)
        for name, builder in sb.BUILDERS.items()
    ]
    checks = (
        data_checks(
            project=project,
            stats_query=sql_input(sb.check_stats(), **values),
            **values,
        )
        .set_display_name("data_checks")
        .set_caching_options(False)
    )
    for b in builds:
        checks.after(b)
    return NamedTuple("Features", **{k: str for k in values})(*[checks.outputs[k] for k in values])


@dsl.pipeline(name="vertex-demo-retail-intelligence")
def retail_pipeline(
    project: str = "",
    location: str = "us-central1",
    run_date: str = "",
    train_automl: bool = False,
    bq_dataset: str = "vertex_demo",
    dashboard_dataset: str = "vertex_demo_dashboard",
):
    finish = record_run(
        project=project,
        location=location,
        job_resource_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        run_id=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        run_date=run_date,
        train_automl=train_automl,
        dashboard_dataset=dashboard_dataset,
        schema_json=schema_json(),
    ).set_display_name("record_run")
    finish.set_caching_options(False)
    with dsl.ExitHandler(finish):
        features = build_features(project=project, location=location, run_date=run_date, bq_dataset=bq_dataset)
        churn = churn_branch(
            project=project,
            location=location,
            bq_dataset=bq_dataset,
            train_automl=train_automl,
            churn_train=features.outputs["churn_train"],
            churn_score=features.outputs["churn_score"],
        )
        clv = clv_branch(
            project=project,
            location=location,
            bq_dataset=bq_dataset,
            clv_train=features.outputs["clv_train"],
            clv_score=features.outputs["clv_score"],
        )
        demand = demand_branch(
            project=project,
            bq_dataset=bq_dataset,
            run_date=features.outputs["run_date"],
            demand_daily=features.outputs["demand_daily"],
        )
        publish_results(
            project=project,
            location=location,
            run_id=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
            run_date=features.outputs["run_date"],
            train_automl=train_automl,
            bq_dataset=bq_dataset,
            dashboard_dataset=dashboard_dataset,
            schema_json=schema_json(),
            churn_model_name=churn.outputs["model_name"],
            churn_version_id=churn.outputs["version_id"],
            churn_evaluation=churn.outputs["evaluation"],
            churn_predictions=churn.outputs["predictions"],
            churn_explanations=churn.outputs["explanations"],
            clv_model_name=clv.outputs["model_name"],
            clv_version_id=clv.outputs["version_id"],
            clv_evaluation=clv.outputs["evaluation"],
            demand_model=demand.outputs["model"],
            demand_training_seconds=demand.outputs["training_seconds"],
            demand_mape=demand.outputs["mape"],
            demand_coverage=demand.outputs["coverage"],
            demand_series=demand.outputs["series"],
            demand_excluded_days=demand.outputs["excluded_days"],
            demand_daily=features.outputs["demand_daily"],
            clv_train_table=features.outputs["clv_train"],
            demand_forecast_table=demand.outputs["forecast_table"],
            demand_holdout_metrics_table=demand.outputs["holdout_metrics_table"],
            demand_holdout_forecast_table=demand.outputs["holdout_forecast_table"],
            actual_days=DEMAND_ACTUAL_DAYS,
            clv_sample_rows=CLV_SAMPLE_ROWS,
            top_customers=TOP_CUSTOMERS,
            churn_target=TARGET_COLUMN,
            churn_negative_class=sb.NEGATIVE_CLASS,
            churn_positive_class=sb.POSITIVE_CLASS,
        ).set_display_name("publish_results").set_caching_options(False)
