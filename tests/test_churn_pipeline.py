import json

import pytest

from pipeline.compile import compile_pipeline

LABEL = {"app": "vertex-demo"}


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    return json.loads(compile_pipeline(tmp_path_factory.mktemp("c") / "pipeline.json").read_text())


def tasks_of(spec, component_prefix):
    """(owning component, task name, task) for every task whose component name starts with the prefix."""
    found = []
    for comp_name, comp in spec["components"].items():
        for name, task in comp.get("dag", {}).get("tasks", {}).items():
            if task["componentRef"]["name"].startswith(component_prefix):
                found.append((comp_name, name, task))
    return found


def constant(task, key):
    return task["inputs"]["parameters"][key]["runtimeValue"]["constant"]


def condition_of(spec, comp_name):
    for comp in spec["components"].values():
        for task in comp.get("dag", {}).get("tasks", {}).values():
            if task["componentRef"]["name"] == comp_name:
                return task.get("triggerPolicy", {}).get("condition")
    return None


def test_training_runs_only_when_train_automl_is_true(spec):
    ((owner, _, _),) = tasks_of(spec, "comp-automl-tabular-training-job")
    assert condition_of(spec, owner) == "inputs.parameter_values['pipelinechannel--train_automl'] == true"
    ((ds_owner, _, _),) = tasks_of(spec, "comp-tabular-dataset-create")
    assert ds_owner == owner


def test_the_other_branch_trains_nothing_and_negates_the_condition(spec):
    gates = {owner: name for owner, name, _ in tasks_of(spec, "comp-churn-gate")}
    assert len(gates) == 2
    train_owner = tasks_of(spec, "comp-automl-tabular-training-job")[0][0]
    (reuse_owner,) = [o for o in gates if o != train_owner]
    assert condition_of(spec, reuse_owner) == ("!(inputs.parameter_values['pipelinechannel--train_automl'] == true)")
    reuse_tasks = spec["components"][reuse_owner]["dag"]["tasks"]
    assert [t["componentRef"]["name"] for t in reuse_tasks.values()] == ["comp-churn-gate-2"]


def test_training_budget_objective_and_target(spec):
    ((_, _, task),) = tasks_of(spec, "comp-automl-tabular-training-job")
    assert constant(task, "budget_milli_node_hours") == 1000
    assert constant(task, "optimization_objective") == "maximize-au-prc"
    assert constant(task, "optimization_prediction_type") == "classification"
    assert constant(task, "target_column") == "outcome"
    assert constant(task, "is_default_version") is False
    cols = constant(task, "column_specs")
    for banned in ("user_id", "feature_cutoff", "source_max_ts", "outcome", "churned"):
        assert banned not in cols


def test_every_job_carries_the_app_label(spec):
    for prefix in (
        "comp-bigquery-query-job",
        "comp-tabular-dataset-create",
        "comp-automl-tabular-training-job",
        "comp-model-batch-predict",
    ):
        found = tasks_of(spec, prefix)
        assert found, prefix
        for _, name, task in found:
            assert constant(task, "labels") == LABEL, name
    ((_, _, task),) = tasks_of(spec, "comp-automl-tabular-training-job")
    assert constant(task, "model_labels") == LABEL


def churn_batch(spec):
    dag = spec["components"]["comp-churn-automl"]["dag"]["tasks"]
    found = {
        constant(t, "job_display_name"): (name, t)
        for name, t in dag.items()
        if t["componentRef"]["name"].startswith("comp-model-batch")
    }
    assert set(found) == {"vertex-demo-churn-batch-predict", "vertex-demo-churn-batch-explain"}
    return found


def test_batch_predictions_have_no_endpoint_and_only_the_second_pass_explains(spec):
    found = churn_batch(spec)
    for key, (_, task) in found.items():
        assert constant(task, "instances_format") == "bigquery"
        assert constant(task, "predictions_format") == "bigquery"
        assert "accelerator_type" not in task["inputs"]["parameters"]
        assert "accelerator_count" not in task["inputs"]["parameters"]
        assert constant(task, "labels") == LABEL
        assert constant(task, "generate_explanation") is key.endswith("explain")
    names = " ".join(spec["components"]).lower()
    assert "endpoint" not in names and "deploy" not in names


def test_pass_one_scores_every_row_and_pass_two_the_top_customers(spec):
    found = churn_batch(spec)
    first_name, first = found["vertex-demo-churn-batch-predict"]
    second_name, second = found["vertex-demo-churn-batch-explain"]
    assert "score_uri" in json.dumps(first["inputs"]["parameters"]["bigquery_source_input_uri"])
    for task in (first, second):
        assert "output_dataset_uri" in json.dumps(task["inputs"]["parameters"]["bigquery_destination_output_uri"])
    assert "churn-explain-input" in json.dumps(second["inputs"]["parameters"]["bigquery_source_input_uri"])
    dag = spec["components"]["comp-churn-automl"]["dag"]["tasks"]
    pick = dag["churn-explain-input"]
    assert first_name in pick["dependentTasks"]
    assert "churn-explain-input" in second["dependentTasks"]
    assert constant(pick, "top_n") == 200 and constant(pick, "at_risk_class") == "churned"


def test_both_passes_use_the_gates_model_version(spec):
    found = churn_batch(spec)
    models = {json.dumps(t["inputs"]["artifacts"]["model"], sort_keys=True) for _, t in found.values()}
    assert len(models) == 1 and "model-get" in models.pop()


def test_churn_outputs_for_publish(spec):
    outputs = spec["components"]["comp-churn-automl"]["dag"]["outputs"]
    assert set(outputs["parameters"]) == {"model_name", "version_id"}
    assert set(outputs["artifacts"]) == {"evaluation", "predictions", "explanations"}


def test_no_task_is_cached(spec):
    leaves = [
        (comp, name, task)
        for comp, c in spec["components"].items()
        for name, task in c.get("dag", {}).get("tasks", {}).items()
        if "dag" not in spec["components"][task["componentRef"]["name"]]
    ]
    assert len(leaves) >= 15
    for comp, name, task in leaves:
        assert not task.get("cachingOptions", {}).get("enableCache"), f"{comp}/{name}"


def test_the_data_check_runs_after_the_builds_and_before_training_gate_and_batch_prediction(spec):
    ((owner, check, _),) = tasks_of(spec, "comp-churn-data-check")
    dag = spec["components"][owner]["dag"]["tasks"]
    builds = [n for n, t in dag.items() if t["componentRef"]["name"].startswith("comp-bigquery-query-job")]
    assert len(builds) == 2 and set(builds) <= set(dag[check]["dependentTasks"])
    assert check in dag["resolve-default-model"]["dependentTasks"]
    assert check in dag["model-get"]["dependentTasks"]
    branches = [t for t in dag.values() if t["componentRef"]["name"].startswith("comp-condition-branches")]
    assert branches and all(check in t["dependentTasks"] for t in branches)


def test_the_explain_input_joins_on_an_integer_user_id_because_batch_output_ids_are_strings(monkeypatch):
    from types import SimpleNamespace

    from google.cloud import bigquery

    from pipeline.churn import churn_explain_input

    sql = []

    class Client:
        def __init__(self, project=None):
            pass

        def query(self, text, job_config=None):
            sql.append((text, job_config.labels))
            return SimpleNamespace(result=lambda: iter([{"n": 200}]))

    monkeypatch.setattr(bigquery, "Client", Client)
    out = churn_explain_input.python_func(
        project="p",
        score_table="`p.d.score`",
        dataset="d",
        predictions=SimpleNamespace(metadata={"projectId": "p", "datasetId": "d", "tableId": "predictions_1"}),
        target_column="outcome",
        at_risk_class="churned",
        top_n=200,
    )
    assert out.rows == 200 and out.uri == "bq://p.d.churn_explain_input"
    assert "SAFE_CAST(user_id AS INT64)" in sql[0][0] and "LIMIT 200" in sql[0][0]
    assert "predicted_outcome" in sql[0][0] and "'churned'" in sql[0][0]
    assert all(labels == {"app": "vertex-demo"} for _, labels in sql)
