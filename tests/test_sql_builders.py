import re

import pytest

from pipeline import sql_builders as sb

VALUES = {"run_date": "2026-11-25", **{t: f"`p.d.{t}`" for t in sb.TABLES}}


def rendered(name):
    return sb.render(sb.BUILDERS[name](), **VALUES)


def features_block(sql):
    start = sql.index("features AS (")
    return sql[start : sql.index("\n)", start)]


@pytest.mark.parametrize("name", sb.TABLES)
def test_renders_without_leftover_tokens(name):
    sql = rendered(name)
    assert not re.search(r"\{\w+\}", sql)
    assert f"CREATE OR REPLACE TABLE `p.d.{name}`" in sql
    assert "'app', 'vertex-demo'" in sql


@pytest.mark.parametrize("name", ["churn_train", "churn_score", "clv_train", "clv_score"])
def test_every_feature_source_is_cut_before_the_cutoff(name):
    sql = rendered(name)
    for alias in ("o", "oi", "e"):
        assert f"{alias}.created_at < cut.ts" in sql
    assert "u.created_at < cut.ts" in sql
    assert ">=" not in features_block(sql)


def test_churn_label_comes_only_from_the_window_after_the_cutoff():
    sql = rendered("churn_train")
    assert "INTERVAL 90 DAY" in sql
    label = sql[sql.index("active AS (") :]
    assert "o.created_at >= cut.ts" in label
    assert "o.created_at < TIMESTAMP(DATE('2026-11-25'))" in label
    assert "churned" not in features_block(sql)


def test_clv_label_uses_spend_after_the_cutoff_without_cancelled_or_returned():
    sql = rendered("clv_train")
    assert "INTERVAL 180 DAY" in sql
    spend = sql[sql.index("spend AS (") :]
    assert "oi.created_at >= cut.ts" in spend
    assert "'Cancelled', 'Returned'" in spend
    assert "ltv_180d" not in features_block(sql)


@pytest.mark.parametrize("name", ["churn_score", "clv_score"])
def test_scoring_uses_run_date_as_cutoff_and_has_no_label(name):
    sql = rendered(name)
    assert "TIMESTAMP(DATE('2026-11-25')) AS ts" in sql
    assert "churned" not in sql and "ltv_180d" not in sql


def test_cancelled_orders_never_count():
    for name in ("churn_train", "clv_train", "demand_daily"):
        assert "'Cancelled'" in rendered(name)


def test_demand_is_zero_filled_per_category_and_ends_before_run_date():
    sql = rendered("demand_daily")
    assert "IFNULL(s.units, 0)" in sql
    assert "DATE_SUB(end_excl, INTERVAL 1 DAY)" in sql
    assert "< bounds.end_excl" in sql


def test_check_stats_covers_every_table_and_reads_cutoff_columns():
    sql = sb.render(sb.check_stats(), **VALUES)
    for t in sb.TABLES:
        assert f"`p.d.{t}`" in sql
    assert "MAX(source_max_ts)" in sql and "MAX(feature_cutoff)" in sql


@pytest.mark.parametrize("name", ["churn_train", "churn_score", "clv_train", "clv_score"])
def test_returns_after_the_cutoff_are_not_features_and_are_checked(name):
    sql = rendered(name)
    assert "oi.returned_at IS NOT NULL AND oi.returned_at < cut.ts" in sql
    assert "MAX(IF(returned, returned_at, NULL)) AS last_return_at" in sql
    assert "IFNULL(bi.last_return_at, bo.last_order_at)" in features_block(sql)


def test_churn_automl_tables_drop_ids_and_cutoffs_and_name_the_minority_class():
    values = {**VALUES, "churn_automl_train": "`p.d.a`", "churn_automl_score": "`p.d.b`"}
    train = sb.render(sb.churn_automl_train(), **values)
    score = sb.render(sb.churn_automl_score(), **values)
    assert "IF(churned = 1, 'churned', 'reorders') AS outcome" in train
    for column in ("user_id", "feature_cutoff", "source_max_ts"):
        assert column not in train.split("FROM")[0]
        assert column not in score.split("FROM")[0].replace("SELECT user_id", "")
    assert "SELECT user_id," in score and "FROM `p.d.churn_score`" in score
    assert not re.search(r"\{\w+\}", train + score)
    assert sb.POSITIVE_CLASS == "reorders"
