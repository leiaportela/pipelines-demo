import json
from types import SimpleNamespace

import pytest

from pipeline.churn import churn_gate

MODEL = "projects/p/locations/us-central1/models/42"


def value(d):
    from google.protobuf import json_format, struct_pb2

    return json_format.ParseDict(d, struct_pb2.Value())


def metrics(auprc, positive=True):
    pts = [
        {
            "confidenceThreshold": 0.3,
            "recall": 0.9,
            "precision": 0.2,
            "falsePositiveRate": 0.4,
            "f1Score": 0.33,
            "truePositiveCount": 9,
            "falsePositiveCount": 36,
            "trueNegativeCount": 54,
            "falseNegativeCount": 1,
        },
        {
            "confidenceThreshold": 0.5,
            "recall": 0.5,
            "precision": 0.4,
            "falsePositiveRate": 0.1,
            "f1Score": 0.44,
            "truePositiveCount": 5,
            "falsePositiveCount": 8,
            "trueNegativeCount": 82,
            "falseNegativeCount": 5,
        },
    ]
    return {"auPrc": auprc, "auRoc": 0.8, "logLoss": 0.3, "confidenceMetrics": pts}


class FakeClient:
    def __init__(self, auprc_by_version, overall_by_version=None):
        self.auprc = auprc_by_version
        self.overall = overall_by_version or {v: 0.99 for v in auprc_by_version}

    empty_calls = 0

    def list_model_evaluations(self, parent):
        if FakeClient.empty_calls:
            FakeClient.empty_calls -= 1
            return []
        version = parent.split("@")[1]
        ev = SimpleNamespace(
            name=f"{parent}/evaluations/1", _pb=SimpleNamespace(metrics=value(metrics(self.overall[version])))
        )
        from google.cloud.aiplatform_v1.types import explanation

        attr = explanation.ModelExplanation(
            mean_attributions=[
                explanation.Attribution(feature_attributions=value({"recency_days": 0.7, "age": 0.1}).struct_value)
            ]
        )
        ev._pb.model_explanation = explanation.ModelExplanation.pb(attr)
        return [ev]

    def list_model_evaluation_slices(self, parent):
        version = parent.split("@")[1].split("/")[0]
        mk = lambda label, auprc: SimpleNamespace(  # noqa: E731
            slice_=SimpleNamespace(dimension="annotationSpec", value=label),
            _pb=SimpleNamespace(metrics=value(metrics(auprc))),
        )
        return [mk("churned", 0.97), mk("reorders", self.auprc[version])]

    def get_model(self, name):
        raise RuntimeError("no training pipeline in tests")


class Registry:
    aliases = []
    versions = []

    def __init__(self, model):
        self.model = model

    def add_version_aliases(self, new_aliases, version):
        Registry.aliases.append((new_aliases, version))

    def list_versions(self):
        return Registry.versions


@pytest.fixture
def run(monkeypatch, tmp_path):
    from google.cloud import aiplatform, aiplatform_v1

    def go(auprc, default=None, trained=None, overall=None):
        Registry.aliases = []
        Registry.versions = [SimpleNamespace(version_id=v) for v in auprc]
        listed = [SimpleNamespace(resource_name=MODEL, version_id=default)] if default else []
        monkeypatch.setattr(aiplatform, "init", lambda **kw: None)
        monkeypatch.setattr(aiplatform.Model, "list", staticmethod(lambda filter=None: listed))
        monkeypatch.setattr(aiplatform, "ModelRegistry", Registry)
        monkeypatch.setattr(aiplatform_v1, "ModelServiceClient", lambda client_options=None: FakeClient(auprc, overall))
        evaluation = SimpleNamespace(path=str(tmp_path / "evaluation.json"))
        result = churn_gate.python_func(
            "p", "us-central1", "vertex-demo-churn", "reorders", evaluation, trained=bool(trained)
        )
        return result, json.loads((tmp_path / "evaluation.json").read_text())

    return go


def test_new_version_with_better_auprc_becomes_default(run):
    result, info = run({"1": 0.30, "2": 0.40}, default="1", trained=True)
    assert result.promoted and result.version_id == "2" and result.versioned_name == "42@2"
    assert Registry.aliases == [(["default"], "2")]
    assert result.auprc == 0.40 and result.previous_auprc == 0.30
    assert info["gate"] == {"promoted": True, "previous_default_auprc": 0.30}


def test_new_version_with_worse_or_equal_auprc_keeps_the_default(run):
    for new in (0.20, 0.30):
        result, info = run({"1": 0.30, "2": new}, default="1", trained=True)
        assert not result.promoted and result.version_id == "1"
        assert Registry.aliases == []
        assert info["version_id"] == "1"


def test_first_version_becomes_default_without_comparison(run):
    result, _ = run({"1": 0.3}, default="1", trained=True)
    assert result.promoted and Registry.aliases == []
    with pytest.raises(RuntimeError, match="no model named"):
        run({"1": 0.3}, default=None, trained=True)


def test_the_newest_version_is_the_one_just_trained(run):
    result, _ = run({"1": 0.30, "2": 0.35, "10": 0.40}, default="1", trained=True)
    assert result.version_id == "10" and result.promoted


def test_positive_class_slice_is_used_not_the_overall_or_majority_metrics(run):
    result, info = run({"1": 0.31}, default="1", overall={"1": 0.99})
    assert result.auprc == 0.31
    assert info["positive_class"] == "reorders" and info["positive_class_slice_found"]
    assert info["overall_auprc"] == 0.99


def test_reuse_without_a_default_fails_with_a_clear_message(run):
    with pytest.raises(RuntimeError, match="No default churn model.*train_automl=true"):
        run({}, default=None)


def test_reuse_writes_the_evaluation_for_publish(run):
    result, info = run({"1": 0.31}, default="1")
    assert not result.promoted and result.model_name == MODEL
    assert info["operating_point"]["threshold"] == 0.5
    assert [p["threshold"] for p in info["pr_curve"]] == [0.3, 0.5]
    assert info["confusion"][1] == {"threshold": 0.5, "tp": 5, "fp": 8, "tn": 82, "fn": 5}
    assert info["feature_importance"][0] == {"feature": "recency_days", "importance": 0.7, "rank": 1}


def test_versioned_name_is_the_short_id_form_that_model_get_expects(run):
    result, _ = run({"1": 0.30}, default="1")
    assert "/" not in result.versioned_name
    assert result.versioned_name == "42@1"


@pytest.mark.parametrize("default_version", ["1", "7"])
def test_gate_output_resolves_through_the_real_model_get_function(run, monkeypatch, default_version):
    get_model_module = pytest.importorskip(
        "google_cloud_pipeline_components.container.v1.model.get_model.remote_runner"
    )
    result, _ = run({default_version: 0.30}, default=default_version)
    requested = []

    class GetClient:
        def __init__(self, client_options=None):
            pass

        def get_model(self, request):
            requested.append(request.name)
            model_id, version = request.name.rsplit("/", 1)[1].split("@")
            return SimpleNamespace(name=f"projects/p/locations/us-central1/models/{model_id}", version_id=version)

    monkeypatch.setattr(get_model_module.aip_v1, "ModelServiceClient", GetClient)
    monkeypatch.setattr(get_model_module.artifact_utils, "update_output_artifacts", lambda *a: None)
    get_model_module.get_model("{}", result.versioned_name, "p", "us-central1")
    assert requested == [f"projects/p/locations/us-central1/models/42@{default_version}"]


def test_the_gate_takes_no_artifact_input_so_any_trained_model_type_works():
    import inspect

    params = inspect.signature(churn_gate.python_func).parameters
    assert params["trained"].annotation is bool
    assert not any("Artifact" in str(p.annotation) and p.name != "evaluation" for p in params.values())


def test_the_trained_gate_runs_after_training_and_takes_no_artifact(tmp_path):
    import json as _json

    from pipeline.compile import compile_pipeline

    spec = _json.loads(compile_pipeline(tmp_path / "pipeline.json").read_text())
    gates = {
        name: task
        for comp in spec["components"].values()
        for name, task in comp.get("dag", {}).get("tasks", {}).items()
        if name.startswith("churn-gate")
    }
    trained, reuse = gates["churn-gate"], gates["churn-gate-2"]
    assert trained["dependentTasks"] == ["automl-tabular-training-job"]
    for task, flag in ((trained, True), (reuse, False)):
        assert not task["inputs"].get("artifacts")
        assert task["inputs"]["parameters"]["trained"]["runtimeValue"]["constant"] is flag


def test_an_evaluation_that_is_still_being_imported_is_waited_for(run, monkeypatch):
    import time

    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    FakeClient.empty_calls = 2
    try:
        result, _ = run({"1": 0.31}, default="1")
    finally:
        FakeClient.empty_calls = 0
    assert result.auprc == 0.31 and len(sleeps) == 2


def test_training_details_are_read_from_the_proto_inputs():
    import inspect

    source = inspect.getsource(churn_gate.python_func)
    assert "pipe._pb.training_task_inputs" in source
