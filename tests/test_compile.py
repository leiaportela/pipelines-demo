import json

from pipeline.compile import compile_pipeline


def test_compiles_with_parameters(tmp_path):
    out = compile_pipeline(tmp_path / "pipeline.json")
    spec = json.loads(out.read_text())
    params = spec["root"]["inputDefinitions"]["parameters"]
    for name in (
        "project",
        "location",
        "run_date",
        "train_automl",
        "bq_dataset",
        "dashboard_dataset",
    ):
        assert name in params
    assert params["train_automl"]["defaultValue"] is False
    assert params["train_automl"]["parameterType"] == "BOOLEAN"
