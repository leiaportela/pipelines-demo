"""Demand branch: BigQuery ML ARIMA_PLUS holdout evaluation, then the production model and a 30-day forecast."""

import re
from typing import NamedTuple

from google_cloud_pipeline_components.v1.bigquery import BigqueryCreateModelJobOp, BigqueryQueryJobOp
from kfp import dsl

LABELS = {"app": "vertex-demo"}
BQ_LOCATION = "US"
HOLDOUT_DAYS = 28
HORIZON_DAYS = 30
CONFIDENCE_LEVEL = 0.9
RAMP_WINDOW_DAYS = 14
RAMP_BASELINE_DAYS = 60
RAMP_FACTOR = 1.5
SERIES_COLUMN = "category"
TIME_COLUMN = "sale_date"
VALUE_COLUMN = "units"
OVERALL = "ALL"
TABLES = (
    "demand_holdout_model",
    "demand_model",
    "demand_holdout_metrics",
    "demand_holdout_forecast",
    "demand_forecast",
)


def sql_input(template: str, **values):
    pieces = [p for p in re.split(r"(\{\w+\})", template) if p]
    return "".join(f"{values[p[1:-1]]}" if p[1:-1] in values else p for p in pieces)


def _end_excl() -> str:
    """First day left out of the demand series (SQL scalar subquery); run date when there is no ramp.

    thelook_ecommerce is synthetic and refreshed continuously; its generator ramps order volume up over the last
    days before the data end (daily totals about 250-420 until 2026-11-20, then 594, 701, 913, 1291, 1874). That
    ramp is a dataset artifact, not demand: left in, ARIMA_PLUS reads it as a trend and the holdout is scored
    against it. The series therefore ends before the first day, within the last RAMP_WINDOW_DAYS days, whose
    total units exceed RAMP_FACTOR times the median daily total of the RAMP_BASELINE_DAYS days before that window.
    The cutoff is detected, not fixed, so it moves with the run date and is empty on data without a ramp.
    """
    run = "DATE('{run_date}')"

    def totals(where: str = "") -> str:
        return (
            f"SELECT {TIME_COLUMN}, SUM({VALUE_COLUMN}) AS total FROM {{demand_daily}} {where} GROUP BY {TIME_COLUMN}"
        )

    baseline = totals(
        f"WHERE {TIME_COLUMN} >= DATE_SUB({run}, INTERVAL {RAMP_BASELINE_DAYS} DAY) "
        f"AND {TIME_COLUMN} < DATE_SUB({run}, INTERVAL {RAMP_WINDOW_DAYS} DAY)"
    )
    return (
        f"(SELECT IFNULL(MIN({TIME_COLUMN}), {run}) FROM ({totals()}) "
        f"WHERE {TIME_COLUMN} >= DATE_SUB({run}, INTERVAL {RAMP_WINDOW_DAYS} DAY) "
        f"AND total > {RAMP_FACTOR} * (SELECT APPROX_QUANTILES(total, 2)[OFFSET(1)] FROM ({baseline})))"
    )


def _cutoff() -> str:
    return f"DATE_SUB({_end_excl()}, INTERVAL {HOLDOUT_DAYS} DAY)"


def excluded_days_sql() -> str:
    return f"SELECT DATE_DIFF(DATE('{{run_date}}'), {_end_excl()}, DAY) AS excluded_days"


def _create_model(model: str, upper_bound: str) -> str:
    return f"""CREATE OR REPLACE MODEL {{{model}}}
OPTIONS (
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = '{TIME_COLUMN}',
  time_series_data_col = '{VALUE_COLUMN}',
  time_series_id_col = '{SERIES_COLUMN}',
  data_frequency = 'DAILY',
  holiday_region = 'US'
) AS
SELECT {SERIES_COLUMN}, {TIME_COLUMN}, {VALUE_COLUMN}
FROM {{demand_daily}}
WHERE {TIME_COLUMN} < {upper_bound}"""


def holdout_model_sql() -> str:
    return _create_model("demand_holdout_model", _cutoff())


def production_model_sql() -> str:
    return _create_model("demand_model", _end_excl())


def _label_model(model: str) -> str:
    """CREATE MODEL reads `labels` as the label column list, so the resource label is set afterwards."""
    return f"ALTER MODEL {{{model}}} SET OPTIONS (labels = [('app', 'vertex-demo')])"


def label_holdout_model_sql() -> str:
    return _label_model("demand_holdout_model")


def label_production_model_sql() -> str:
    return _label_model("demand_model")


def holdout_forecast_sql() -> str:
    return f"""CREATE OR REPLACE TABLE {{demand_holdout_forecast}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
SELECT
  f.{SERIES_COLUMN} AS category,
  DATE(f.forecast_timestamp) AS date,
  a.{VALUE_COLUMN} AS units,
  GREATEST(0, f.forecast_value) AS forecast_units,
  GREATEST(0, f.prediction_interval_lower_bound) AS lower_bound,
  f.prediction_interval_upper_bound AS upper_bound
FROM ML.FORECAST(
  MODEL {{demand_holdout_model}}, STRUCT({HOLDOUT_DAYS} AS horizon, {CONFIDENCE_LEVEL} AS confidence_level)
) AS f
JOIN {{demand_daily}} AS a
  ON a.{SERIES_COLUMN} = f.{SERIES_COLUMN} AND a.{TIME_COLUMN} = DATE(f.forecast_timestamp)
WHERE a.{TIME_COLUMN} >= {_cutoff()} AND a.{TIME_COLUMN} < {_end_excl()}"""


def holdout_metrics_sql() -> str:
    """MAPE over days with actual units above zero; coverage over all holdout days."""
    return f"""CREATE OR REPLACE TABLE {{demand_holdout_metrics}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH per_day AS (
  SELECT category, units, forecast_units,
         IF(units > 0, ABS(units - forecast_units) / units, NULL) AS ape,
         units BETWEEN lower_bound AND upper_bound AS covered
  FROM {{demand_holdout_forecast}}
)
SELECT category, AVG(ape) AS mape, AVG(IF(covered, 1.0, 0.0)) AS coverage, COUNT(*) AS holdout_days
FROM per_day GROUP BY category
UNION ALL
SELECT '{OVERALL}', AVG(ape), AVG(IF(covered, 1.0, 0.0)), COUNT(*)
FROM per_day"""


def forecast_sql() -> str:
    """The production model ends before the ramp, so it forecasts across the gap and the table keeps the 30 days
    from the run date (the gap is at most RAMP_WINDOW_DAYS days)."""
    return f"""CREATE OR REPLACE TABLE {{demand_forecast}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
SELECT
  {SERIES_COLUMN} AS category,
  DATE(forecast_timestamp) AS date,
  GREATEST(0, forecast_value) AS forecast,
  GREATEST(0, prediction_interval_lower_bound) AS lower,
  prediction_interval_upper_bound AS upper
FROM ML.FORECAST(
  MODEL {{demand_model}}, STRUCT({HORIZON_DAYS + RAMP_WINDOW_DAYS} AS horizon, {CONFIDENCE_LEVEL} AS confidence_level)
)
WHERE DATE(forecast_timestamp) >= DATE('{{run_date}}')
  AND DATE(forecast_timestamp) < DATE_ADD(DATE('{{run_date}}'), INTERVAL {HORIZON_DAYS} DAY)"""


@dsl.component(base_image="python:3.12-slim")
def forecast_names(project: str, dataset: str) -> NamedTuple(
    "Names",
    demand_holdout_model=str,
    demand_model=str,
    demand_holdout_metrics=str,
    demand_holdout_forecast=str,
    demand_forecast=str,
):
    from collections import namedtuple

    tables = (
        "demand_holdout_model",
        "demand_model",
        "demand_holdout_metrics",
        "demand_holdout_forecast",
        "demand_forecast",
    )
    out = namedtuple("Names", tables)
    return out(*[f"`{project}.{dataset}.{t}`" for t in tables])


@dsl.component(base_image="python:3.12-slim", packages_to_install=["google-cloud-bigquery==3.47.0"])
def forecast_summary(
    project: str,
    run_date: str,
    holdout_days: int,
    horizon_days: int,
    demand_daily: str,
    model_id: str,
    holdout_model_id: str,
    forecast_table: str,
    metrics_table: str,
    holdout_forecast_table: str,
    model_job: str,
    holdout_job: str,
    excluded_days_query: str,
) -> NamedTuple(
    "Summary",
    forecast_table=str,
    holdout_metrics_table=str,
    holdout_forecast_table=str,
    model=str,
    holdout_model=str,
    training_seconds=float,
    mape=float,
    coverage=float,
    series=int,
    excluded_days=int,
):
    """Check the forecast has every series and day, and read the overall holdout metrics and training time."""
    import datetime
    import json
    from collections import namedtuple

    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    labels = bigquery.QueryJobConfig(labels={"app": "vertex-demo"})

    def rows(sql):
        return [dict(r.items()) for r in client.query(sql, job_config=labels).result()]

    stat = rows(
        f"SELECT COUNT(*) AS n, COUNT(DISTINCT category) AS s, MIN(date) AS d0, MAX(date) AS d1 FROM {forecast_table}"
    )[0]
    series = rows(f"SELECT COUNT(DISTINCT category) AS n FROM {demand_daily}")[0]["n"]
    excluded = int(rows(excluded_days_query)[0]["excluded_days"])
    start = datetime.date.fromisoformat(run_date)
    problems = []
    if stat["s"] != series:
        problems.append(f"forecast has {stat['s']} series, expected {series}")
    if stat["n"] != series * horizon_days:
        problems.append(f"forecast has {stat['n']} rows, expected {series * horizon_days}")
    if stat["d0"] != start or stat["d1"] != start + datetime.timedelta(days=horizon_days - 1):
        problems.append(f"forecast covers {stat['d0']}..{stat['d1']}, expected {start} for {horizon_days} days")
    overall = rows(f"SELECT mape, coverage, holdout_days FROM {metrics_table} WHERE category = 'ALL'")
    per_category = rows(f"SELECT COUNT(*) AS n FROM {metrics_table} WHERE category != 'ALL'")[0]["n"]
    if per_category != series:
        problems.append(f"holdout metrics for {per_category} categories, expected {series}")
    if not overall or overall[0]["holdout_days"] != series * holdout_days:
        problems.append(f"holdout has {overall[0]['holdout_days'] if overall else 0} series-days")
    if problems:
        raise RuntimeError("forecast checks failed: " + "; ".join(problems))

    def seconds(gcp_resources):
        try:
            uri = json.loads(gcp_resources)["resources"][0]["resourceUri"]
            job_id = uri.split("/jobs/")[1].split("?")[0]
            location = uri.split("location=")[1] if "location=" in uri else "US"
            job = client.get_job(job_id, location=location)
            return (job.ended - job.started).total_seconds()
        except Exception:
            return -1.0

    parts = [seconds(model_job), seconds(holdout_job)]
    total = -1.0 if min(parts) < 0 else sum(parts)
    out = namedtuple(
        "Summary",
        "forecast_table holdout_metrics_table holdout_forecast_table model holdout_model "
        "training_seconds mape coverage series excluded_days",
    )
    return out(
        forecast_table,
        metrics_table,
        holdout_forecast_table,
        model_id,
        holdout_model_id,
        total,
        float(overall[0]["mape"]),
        float(overall[0]["coverage"]),
        int(stat["s"]),
        excluded,
    )


@dsl.pipeline(name="demand-bqml-arima")
def demand_branch(project: str, bq_dataset: str, run_date: str, demand_daily: str) -> NamedTuple(
    "Demand",
    forecast_table=str,
    holdout_metrics_table=str,
    holdout_forecast_table=str,
    model=str,
    holdout_model=str,
    training_seconds=float,
    mape=float,
    coverage=float,
    series=int,
    excluded_days=int,
):
    names = forecast_names(project=project, dataset=bq_dataset).set_caching_options(False)
    values = {"run_date": run_date, "demand_daily": demand_daily, **{t: names.outputs[t] for t in TABLES}}

    def create_model(builder, display):
        return (
            BigqueryCreateModelJobOp(
                query=sql_input(builder(), **values), project=project, location=BQ_LOCATION, labels=LABELS
            )
            .set_display_name(display)
            .set_caching_options(False)
        )

    def query(builder, display):
        return (
            BigqueryQueryJobOp(
                query=sql_input(builder(), **values), project=project, location=BQ_LOCATION, labels=LABELS
            )
            .set_display_name(display)
            .set_caching_options(False)
        )

    holdout_model = create_model(holdout_model_sql, "train_holdout_arima_plus")
    label_holdout = query(label_holdout_model_sql, "label_holdout_model").after(holdout_model)
    holdout_forecast = query(holdout_forecast_sql, "forecast_holdout").after(label_holdout)
    metrics = query(holdout_metrics_sql, "evaluate_holdout").after(holdout_forecast)
    model = create_model(production_model_sql, "train_production_arima_plus").after(metrics)
    label_production = query(label_production_model_sql, "label_production_model").after(model)
    forecast = query(forecast_sql, "forecast_30_days").after(label_production)
    summary = (
        forecast_summary(
            project=project,
            run_date=run_date,
            holdout_days=HOLDOUT_DAYS,
            horizon_days=HORIZON_DAYS,
            demand_daily=demand_daily,
            model_id=names.outputs["demand_model"],
            holdout_model_id=names.outputs["demand_holdout_model"],
            forecast_table=names.outputs["demand_forecast"],
            metrics_table=names.outputs["demand_holdout_metrics"],
            holdout_forecast_table=names.outputs["demand_holdout_forecast"],
            model_job=model.outputs["gcp_resources"],
            holdout_job=holdout_model.outputs["gcp_resources"],
            excluded_days_query=sql_input(excluded_days_sql(), **values),
        )
        .set_display_name("check_forecast")
        .set_caching_options(False)
        .after(forecast)
    )
    return NamedTuple(
        "Demand",
        forecast_table=str,
        holdout_metrics_table=str,
        holdout_forecast_table=str,
        model=str,
        holdout_model=str,
        training_seconds=float,
        mape=float,
        coverage=float,
        series=int,
        excluded_days=int,
    )(
        summary.outputs["forecast_table"],
        summary.outputs["holdout_metrics_table"],
        summary.outputs["holdout_forecast_table"],
        summary.outputs["model"],
        summary.outputs["holdout_model"],
        summary.outputs["training_seconds"],
        summary.outputs["mape"],
        summary.outputs["coverage"],
        summary.outputs["series"],
        summary.outputs["excluded_days"],
    )
