import inspect
import json
import re

import pytest

from pipeline import clv
from pipeline import sql_builders as sb
from pipeline.compile import compile_pipeline

LABEL = {"app": "vertex-demo"}
VALUES = {
    "clv_train": "`p.d.clv_train`",
    "clv_score": "`p.d.clv_score`",
    "clv_model_train": "`p.d.clv_model_train`",
    "clv_model_score": "`p.d.clv_model_score`",
}


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    return json.loads(compile_pipeline(tmp_path_factory.mktemp("c") / "pipeline.json").read_text())


def tasks_of(spec, prefix):
    return [
        t
        for comp in spec["components"].values()
        for t in comp.get("dag", {}).get("tasks", {}).values()
        if t["componentRef"]["name"].startswith(prefix)
    ]


def branch_tasks(spec, branch, prefix):
    dag = spec["components"][f"comp-{branch}"]["dag"]["tasks"]
    return [t for t in dag.values() if t["componentRef"]["name"].startswith(prefix)]


def constant(task, key):
    return task["inputs"]["parameters"][key]["runtimeValue"]["constant"]


def executor_source(spec, prefix):
    return next(ex["container"]["command"][-1] for n, ex in spec["deploymentSpec"]["executors"].items() if prefix in n)


def test_tuning_is_capped_at_12_trials_3_in_parallel_on_cpu(spec):
    (tune,) = tasks_of(spec, "comp-clv-tune")
    assert constant(tune, "max_trial_count") == 12
    assert constant(tune, "parallel_trial_count") == 3
    assert constant(tune, "machine_type") == "n1-standard-4"
    source = executor_source(spec, "clv-tune")
    assert "accelerator" not in source.lower() and "gpu" not in source.lower()
    assert 'metric_spec={"rmse": "minimize"}' in source
    assert source.count("labels=labels") == 2 and 'labels = {"app": "vertex-demo"}' in source


def test_trainer_and_serving_images_are_pinned_cpu_xgboost(spec):
    (tune,) = tasks_of(spec, "comp-clv-tune")
    assert re.fullmatch(r".*/training/xgboost-cpu\.[\d-]+@sha256:[0-9a-f]{64}", constant(tune, "train_image"))
    (register,) = tasks_of(spec, "comp-clv-register")
    assert re.fullmatch(r".*/prediction/xgboost-cpu\.[\d-]+:[\w.]+", constant(register, "serving_image"))


def test_every_job_carries_the_app_label(spec):
    builds = branch_tasks(spec, "clv-xgboost", "comp-bigquery-query-job")
    assert len(builds) == 3  # two table builds and the scoring export
    for task in builds:
        assert constant(task, "labels") == LABEL
    (batch,) = branch_tasks(spec, "clv-xgboost", "comp-model-batch-predict")
    assert constant(batch, "labels") == LABEL
    assert 'labels = {"app": "vertex-demo"}' in executor_source(spec, "clv-register")
    assert "labels=labels" in executor_source(spec, "clv-register")


def test_batch_prediction_is_cpu_only_without_an_endpoint(spec):
    (batch,) = branch_tasks(spec, "clv-xgboost", "comp-model-batch-predict")
    params = batch["inputs"]["parameters"]
    assert "accelerator_type" not in params and "accelerator_count" not in params
    names = " ".join(spec["components"]).lower()
    assert "endpoint" not in names and "deploy" not in names
    for source in (executor_source(spec, "clv-tune"), executor_source(spec, "clv-register")):
        assert "endpoint" not in source.lower() and ".deploy(" not in source


def test_batch_prediction_sends_arrays_in_the_trainers_feature_order(spec):
    (batch,) = branch_tasks(spec, "clv-xgboost", "comp-model-batch-predict")
    assert constant(batch, "instance_type") == "array"
    assert constant(batch, "key_field") == "user_id"
    assert constant(batch, "included_fields") == list(sb.CLV_MODEL_FEATURES)
    (tune,) = tasks_of(spec, "comp-clv-tune")
    assert constant(tune, "features").split(",") == list(sb.CLV_MODEL_FEATURES)


def test_batch_prediction_uses_jsonl_because_bigquery_output_refuses_a_key_field(spec):
    (batch,) = branch_tasks(spec, "clv-xgboost", "comp-model-batch-predict")
    assert constant(batch, "instances_format") == "jsonl"
    assert constant(batch, "predictions_format") == "jsonl"
    assert "bigquery_source_input_uri" not in batch["inputs"]["parameters"] or not constant(
        batch, "bigquery_source_input_uri"
    )
    export = sb.render(sb.clv_score_export(), clv_export_uri="gs://b/x", clv_model_score="`p.d.s`")
    assert export.startswith("EXPORT DATA OPTIONS (uri='gs://b/x/part-*.jsonl', format='JSON'")
    assert export.endswith("SELECT * FROM `p.d.s`")


def test_predictions_are_loaded_into_bigquery_with_a_row_count_check(spec):
    (load,) = tasks_of(spec, "comp-clv-load-predictions")
    assert 'labels={"app": "vertex-demo"}' in executor_source(spec, "clv-load-predictions")
    assert "check.users != expected" in executor_source(spec, "clv-load-predictions")
    assert 'SchemaField("user_id", "INT64")' in executor_source(spec, "clv-load-predictions")


def test_scoring_table_columns_match_the_feature_list():
    score = sb.clv_model_score()
    train = sb.clv_model_train()
    aliases = lambda q: re.findall(r"\bAS (?!FLOAT64)(\w+)", q)  # noqa: E731
    assert aliases(score) == list(sb.CLV_MODEL_FEATURES)
    assert aliases(train) == list(sb.CLV_MODEL_FEATURES)
    assert "user_id" in score.split("FROM")[0] and "ltv_180d" in train.split("FROM")[0]
    assert "ltv_180d" not in score


def test_training_table_reads_only_clv_train_and_score_only_clv_score():
    assert sb.clv_model_train().rstrip().endswith("FROM {clv_train}")
    assert sb.clv_model_score().rstrip().endswith("FROM {clv_score}")
    for builder in (sb.clv_model_train, sb.clv_model_score):
        assert "labels=[('app', 'vertex-demo')]" in builder()


def test_model_is_registered_as_a_labelled_default_version_of_one_stable_model():
    source = inspect.getsource(clv.clv_register.python_func)
    assert "parent_model=existing[0].resource_name if existing else None" in source
    assert "is_default_version=True" in source
    assert clv.MODEL_DISPLAY_NAME == "vertex-demo-clv"


def test_register_gets_the_short_model_ref_and_the_branch_outputs_what_publish_needs(spec):
    (get,) = branch_tasks(spec, "clv-xgboost", "comp-model-get")
    assert "versioned_name" in json.dumps(get["inputs"])
    outputs = spec["components"]["comp-clv-xgboost"]["dag"]["outputs"]
    assert set(outputs["parameters"]) == {"model_name", "version_id"}
    assert set(outputs["artifacts"]) == {"evaluation", "predictions"}


def test_no_clv_task_is_cached(spec):
    for prefix in ("comp-clv-tune", "comp-clv-register", "comp-clv-inputs", "comp-model-get"):
        for task in tasks_of(spec, prefix):
            assert task["cachingOptions"].get("enableCache") is not True


def test_the_full_pipeline_runs_the_clv_branch_regardless_of_train_automl(spec):
    root = spec["components"]["comp-exit-handler-1"]["dag"]["tasks"]
    (task,) = [t for t in root.values() if t["componentRef"]["name"] == "comp-clv-xgboost"]
    assert "triggerPolicy" not in task
    assert "train_automl" not in json.dumps(task["inputs"])


def test_component_sources_run_standalone_in_the_container(spec):
    for name, ex in spec["deploymentSpec"]["executors"].items():
        if "clv" in name:
            source = ex["container"]["command"][-1]
            exec(compile(source, name, "exec"), {})
