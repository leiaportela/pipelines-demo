"""SQL builders for the feature, label and demand tables.

Source: bigquery-public-data.thelook_ecommerce (synthetic, refreshed continuously).
Templates use the tokens {run_date} and the table names {churn_train}, {churn_score}, {clv_train},
{clv_score}, {demand_daily}; `render` fills them with strings and the pipeline fills them with task outputs.

Business rules:
- Cancelled orders and items never count: the dataset keeps only their final status, not when they were
  cancelled, so they cannot be placed on a timeline.
- Returned items count as spend until `returned_at`, so a feature at a cutoff only knows returns that already
  happened. Labels are measured after the fact: CLV spend excludes items that were ever returned.
- Everything is cut at the run date: rows created on or after it are dropped, which also drops timestamps
  that lie in the future.
- Churn: a customer with at least one order before the cutoff (run date - 90 days) who places no order in
  [cutoff, run date). CLV: spend in [cutoff, run date) with cutoff = run date - 180 days.
- Scoring tables use the same features with the run date as cutoff.
- Demand: daily units per product category over the last 730 days before the run date, zero-filled.
"""

import re

SRC = "`bigquery-public-data.thelook_ecommerce`"
CHURN_WINDOW_DAYS = 90
CLV_WINDOW_DAYS = 180
DEMAND_HISTORY_DAYS = 730
EVENT_LOOKBACK_DAYS = 90

TABLES = ("churn_train", "churn_score", "clv_train", "clv_score", "demand_daily")
TOKENS = ("run_date", *TABLES)


def _cutoff_date(days: int) -> str:
    return f"DATE_SUB(DATE('{{run_date}}'), INTERVAL {days} DAY)"


def _features(cutoff: str) -> str:
    """CTEs ending in `features`, one row per customer, using only data created before `cutoff` (a DATE expression)."""
    return f"""
cut AS (SELECT TIMESTAMP({cutoff}) AS ts, {cutoff} AS d),
ord_pre AS (
  SELECT o.user_id, o.order_id, o.created_at
  FROM {SRC}.orders AS o, cut
  WHERE o.status != 'Cancelled' AND o.created_at < cut.ts
),
item_pre AS (
  SELECT oi.user_id, oi.created_at, oi.returned_at, oi.sale_price, p.category,
         (oi.returned_at IS NOT NULL AND oi.returned_at < cut.ts) AS returned
  FROM {SRC}.order_items AS oi
  JOIN {SRC}.products AS p ON p.id = oi.product_id, cut
  WHERE oi.status != 'Cancelled' AND oi.created_at < cut.ts
),
by_order AS (
  SELECT user_id, COUNT(DISTINCT order_id) AS frequency, MAX(created_at) AS last_order_at,
         MIN(created_at) AS first_order_at
  FROM ord_pre GROUP BY user_id
),
by_item AS (
  SELECT user_id, COUNT(*) AS items, SUM(IF(returned, 0, sale_price)) AS monetary,
         COUNTIF(returned) AS return_count, COUNT(DISTINCT category) AS n_categories,
         MAX(created_at) AS last_item_at, MAX(IF(returned, returned_at, NULL)) AS last_return_at
  FROM item_pre GROUP BY user_id
),
top_cat AS (
  SELECT user_id, ARRAY_AGG(category ORDER BY n DESC, category LIMIT 1)[OFFSET(0)] AS top_category
  FROM (SELECT user_id, category, COUNT(*) AS n FROM item_pre GROUP BY user_id, category)
  GROUP BY user_id
),
ev AS (
  SELECT e.user_id, COUNT(*) AS events_90d, COUNT(DISTINCT e.session_id) AS sessions_90d,
         MAX(e.created_at) AS last_event_at
  FROM {SRC}.events AS e, cut
  WHERE e.user_id IS NOT NULL AND e.created_at < cut.ts
    AND e.created_at >= TIMESTAMP_SUB(cut.ts, INTERVAL {EVENT_LOOKBACK_DAYS} DAY)
  GROUP BY e.user_id
),
features AS (
  SELECT
    u.id AS user_id,
    u.age, u.gender, u.country, u.traffic_source,
    DATE_DIFF(cut.d, DATE(u.created_at), DAY) AS tenure_days,
    DATE_DIFF(cut.d, DATE(bo.last_order_at), DAY) AS recency_days,
    DATE_DIFF(DATE(bo.last_order_at), DATE(bo.first_order_at), DAY) AS order_span_days,
    bo.frequency,
    bi.items,
    IFNULL(bi.monetary, 0) AS monetary,
    SAFE_DIVIDE(IFNULL(bi.monetary, 0), bo.frequency) AS avg_order_value,
    bi.return_count,
    SAFE_DIVIDE(bi.return_count, bi.items) AS return_rate,
    bi.n_categories,
    tc.top_category,
    IFNULL(ev.events_90d, 0) AS events_90d,
    IFNULL(ev.sessions_90d, 0) AS sessions_90d,
    DATE_DIFF(cut.d, DATE(ev.last_event_at), DAY) AS days_since_last_event,
    cut.ts AS feature_cutoff,
    GREATEST(bo.last_order_at, IFNULL(bi.last_item_at, bo.last_order_at),
             IFNULL(bi.last_return_at, bo.last_order_at), IFNULL(ev.last_event_at, bo.last_order_at),
             u.created_at) AS source_max_ts
  FROM by_order AS bo
  JOIN {SRC}.users AS u ON u.id = bo.user_id
  JOIN by_item AS bi USING (user_id)
  LEFT JOIN top_cat AS tc USING (user_id)
  LEFT JOIN ev USING (user_id), cut
  WHERE u.created_at < cut.ts
)"""


def churn_train() -> str:
    cutoff = _cutoff_date(CHURN_WINDOW_DAYS)
    return f"""CREATE OR REPLACE TABLE {{churn_train}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH {_features(cutoff).lstrip()},
active AS (
  SELECT DISTINCT o.user_id
  FROM {SRC}.orders AS o, cut
  WHERE o.status != 'Cancelled' AND o.created_at >= cut.ts AND o.created_at < TIMESTAMP(DATE('{{run_date}}'))
)
SELECT f.*, IF(a.user_id IS NULL, 1, 0) AS churned
FROM features AS f LEFT JOIN active AS a USING (user_id)"""


def churn_score() -> str:
    return f"""CREATE OR REPLACE TABLE {{churn_score}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH {_features("DATE('{run_date}')").lstrip()}
SELECT * FROM features"""


def clv_train() -> str:
    cutoff = _cutoff_date(CLV_WINDOW_DAYS)
    return f"""CREATE OR REPLACE TABLE {{clv_train}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH {_features(cutoff).lstrip()},
spend AS (
  SELECT oi.user_id, SUM(oi.sale_price) AS ltv_180d
  FROM {SRC}.order_items AS oi, cut
  WHERE oi.status NOT IN ('Cancelled', 'Returned') AND oi.returned_at IS NULL
    AND oi.created_at >= cut.ts AND oi.created_at < TIMESTAMP(DATE('{{run_date}}'))
  GROUP BY oi.user_id
)
SELECT f.*, IFNULL(s.ltv_180d, 0) AS ltv_180d
FROM features AS f LEFT JOIN spend AS s USING (user_id)"""


def clv_score() -> str:
    return f"""CREATE OR REPLACE TABLE {{clv_score}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH {_features("DATE('{run_date}')").lstrip()}
SELECT * FROM features"""


def demand_daily() -> str:
    return f"""CREATE OR REPLACE TABLE {{demand_daily}} OPTIONS (labels=[('app', 'vertex-demo')]) AS
WITH bounds AS (
  SELECT DATE('{{run_date}}') AS end_excl,
         DATE_SUB(DATE('{{run_date}}'), INTERVAL {DEMAND_HISTORY_DAYS} DAY) AS start_d
),
cats AS (SELECT DISTINCT category FROM {SRC}.products WHERE category IS NOT NULL),
days AS (
  SELECT d AS sale_date
  FROM bounds, UNNEST(GENERATE_DATE_ARRAY(start_d, DATE_SUB(end_excl, INTERVAL 1 DAY))) AS d
),
sold AS (
  SELECT p.category, DATE(oi.created_at) AS sale_date, COUNT(*) AS units
  FROM {SRC}.order_items AS oi
  JOIN {SRC}.products AS p ON p.id = oi.product_id, bounds
  WHERE oi.status != 'Cancelled' AND DATE(oi.created_at) >= bounds.start_d AND DATE(oi.created_at) < bounds.end_excl
  GROUP BY 1, 2
)
SELECT c.category, d.sale_date, IFNULL(s.units, 0) AS units
FROM cats AS c CROSS JOIN days AS d LEFT JOIN sold AS s USING (category, sale_date)"""


CHURN_CATEGORICAL = ("gender", "country", "traffic_source", "top_category")
CHURN_NUMERIC = (
    "age",
    "tenure_days",
    "recency_days",
    "order_span_days",
    "frequency",
    "items",
    "monetary",
    "avg_order_value",
    "return_count",
    "return_rate",
    "n_categories",
    "events_90d",
    "sessions_90d",
    "days_since_last_event",
)
CHURN_FEATURES = CHURN_CATEGORICAL + CHURN_NUMERIC
POSITIVE_CLASS = "reorders"
NEGATIVE_CLASS = "churned"


CHURN_UNKNOWN_CATEGORY = "unknown"
# Value that replaces NULL, per numeric column. AutoML drops every row with a NULL in a numeric column, and most
# customers have no events, so days_since_last_event is NULL for about 85% of rows.
# - days_since_last_event: lookback window + 1, meaning "no event inside the window", beyond any observed value
# - age: -1, a value no real customer has
# - counts, sums and rates: 0, which is what an absent activity measures
CHURN_NULL_DEFAULTS = {
    **{c: 0 for c in CHURN_NUMERIC},
    "age": -1,
    "days_since_last_event": EVENT_LOOKBACK_DAYS + 1,
}


def _churn_automl_columns() -> str:
    categorical = [f"COALESCE({c}, '{CHURN_UNKNOWN_CATEGORY}') AS {c}" for c in CHURN_CATEGORICAL]
    numeric = [f"COALESCE({c}, {CHURN_NULL_DEFAULTS[c]}) AS {c}" for c in CHURN_NUMERIC]
    return ", ".join([*categorical, *numeric])


def churn_automl_train() -> str:
    """AutoML input: the string target `outcome` and the feature columns only (no id, cutoff or source timestamp).

    Every feature is wrapped in COALESCE (see CHURN_NULL_DEFAULTS) so no feature column is NULL.

    The positive class is the minority one, customers who order again ("reorders", about 7%): AutoML's
    maximize-au-prc objective targets the less common class, and the AUC-PR gate and curves use it too.
    """
    cols = _churn_automl_columns()
    return (
        "CREATE OR REPLACE TABLE {churn_automl_train} OPTIONS (labels=[('app', 'vertex-demo')]) AS\n"
        f"SELECT IF(churned = 1, '{NEGATIVE_CLASS}', '{POSITIVE_CLASS}') AS outcome, {cols}\n"
        "FROM {churn_train}"
    )


def churn_automl_score() -> str:
    cols = _churn_automl_columns()
    return (
        "CREATE OR REPLACE TABLE {churn_automl_score} OPTIONS (labels=[('app', 'vertex-demo')]) AS\n"
        f"SELECT user_id, {cols}\nFROM {{churn_score}}"
    )


MAX_INVALID_FRACTION = 0.005


def automl_check_sql(table_token: str, with_target: bool) -> str:
    """One row: the row count, per feature column the rows AutoML would reject, and the NULL targets.

    Numeric columns count NULL, NaN and Inf; categorical columns count NULL. `table_token` is a {placeholder}.
    """
    parts = ["COUNT(*) AS n_rows"]
    for c in CHURN_NUMERIC:
        parts.append(
            f"COUNTIF({c} IS NULL OR IS_NAN(CAST({c} AS FLOAT64)) OR IS_INF(CAST({c} AS FLOAT64))) AS invalid__{c}"
        )
    parts += [f"COUNTIF({c} IS NULL) AS invalid__{c}" for c in CHURN_CATEGORICAL]
    if with_target:
        parts.append("COUNTIF(outcome IS NULL) AS null_target")
    return "SELECT " + ", ".join(parts) + f" FROM {table_token}"


def automl_train_check() -> str:
    return automl_check_sql("{churn_automl_train}", with_target=True)


def automl_score_check() -> str:
    return automl_check_sql("{churn_automl_score}", with_target=False)


CLV_TARGET = "ltv_180d"
CLV_TRAFFIC_SOURCES = ("Organic", "Search", "Display", "Facebook", "Email")
CLV_NUMERIC = tuple(c for c in CHURN_NUMERIC)
CLV_MODEL_FEATURES = (
    *CLV_NUMERIC,
    "is_female",
    *[f"traffic_{t.lower()}" for t in CLV_TRAFFIC_SOURCES],
)


def _clv_model_columns() -> str:
    """Numeric-only feature expressions, in the order the XGBoost model and batch prediction both use.

    Missing values become -1 in training and scoring alike; country and top category are left out because the
    prebuilt XGBoost container only takes numeric arrays.
    """
    numeric = [f"IFNULL(CAST({c} AS FLOAT64), -1) AS {c}" for c in CLV_NUMERIC]
    gender = ["IF(gender = 'F', 1.0, 0.0) AS is_female"]
    traffic = [f"IF(traffic_source = '{t}', 1.0, 0.0) AS traffic_{t.lower()}" for t in CLV_TRAFFIC_SOURCES]
    return ",\n  ".join([*numeric, *gender, *traffic])


def clv_model_train() -> str:
    return (
        "CREATE OR REPLACE TABLE {clv_model_train} OPTIONS (labels=[('app', 'vertex-demo')]) AS\n"
        f"SELECT user_id, {CLV_TARGET},\n  {_clv_model_columns()}\nFROM {{clv_train}}"
    )


def clv_model_score() -> str:
    return (
        "CREATE OR REPLACE TABLE {clv_model_score} OPTIONS (labels=[('app', 'vertex-demo')]) AS\n"
        f"SELECT user_id,\n  {_clv_model_columns()}\nFROM {{clv_score}}"
    )


def clv_score_export() -> str:
    """Export the scoring table as JSONL: the batch prediction needs a key field, which BigQuery output refuses."""
    return (
        "EXPORT DATA OPTIONS (uri='{clv_export_uri}/part-*.jsonl', format='JSON', overwrite=true) AS\n"
        "SELECT * FROM {clv_model_score}"
    )


CLV_BUILDERS = {"clv_model_train": clv_model_train, "clv_model_score": clv_model_score}


CHURN_BUILDERS = {
    "churn_automl_train": churn_automl_train,
    "churn_automl_score": churn_automl_score,
}

BUILDERS = {
    "churn_train": churn_train,
    "churn_score": churn_score,
    "clv_train": clv_train,
    "clv_score": clv_score,
    "demand_daily": demand_daily,
}


def check_stats() -> str:
    """One row per table: name, row count, newest source timestamp, cutoff and label rate (NULL when unlabeled)."""
    parts = [
        f"SELECT '{name}' AS table_name, COUNT(*) AS n_rows, MAX(source_max_ts) AS max_source_ts, "
        f"MAX(feature_cutoff) AS cutoff, {label} AS label_rate, NULL AS n_series FROM {{{name}}}"
        for name, label in (
            ("churn_train", "AVG(churned)"),
            ("churn_score", "NULL"),
            ("clv_train", "AVG(IF(ltv_180d > 0, 1, 0))"),
            ("clv_score", "NULL"),
        )
    ]
    parts.append(
        "SELECT 'demand_daily', COUNT(*), TIMESTAMP(MAX(sale_date)), TIMESTAMP(DATE('{run_date}')), NULL, "
        "COUNT(DISTINCT category) FROM {demand_daily}"
    )
    return "\nUNION ALL\n".join(parts)


def render(template: str, **values: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)] if m.group(1) in values else m.group(0), template)
