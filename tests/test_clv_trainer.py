import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

import xgboost as xgb  # noqa: E402
from clv_trainer import task  # noqa: E402
from clv_trainer.metrics import regression_metrics  # noqa: E402

from pipeline.sql_builders import CLV_MODEL_FEATURES  # noqa: E402

FEATURES = list(CLV_MODEL_FEATURES)


def synthetic(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "user_id", np.arange(1, n + 1))
    frame.insert(1, "ltv_180d", 5 * frame[FEATURES[0]] + 2 * frame[FEATURES[1]] + rng.normal(scale=0.5, size=n))
    return frame


def args_for(tmp_path, **extra):
    argv = [
        "--project=p",
        "--train_table=p.d.t",
        f"--features={','.join(FEATURES)}",
        f"--model_dir={tmp_path / 'trial' / 'model'}",
        "--n_estimators=60",
        "--max_depth=3",
    ]
    argv += [f"--{k}={v}" for k, v in extra.items()]
    return task.parse_args(argv)


def test_metrics_on_known_values():
    m = regression_metrics([0, 0, 10, 10], [1, -1, 8, 12])
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))
    assert m["mae"] == pytest.approx(1.5)
    assert m["bias"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1 - 10 / 100)


def test_split_is_reproducible_disjoint_and_about_a_fifth():
    frame = synthetic(5000)
    fit1, valid1 = task.split(frame, "user_id")
    fit2, valid2 = task.split(frame.sample(frac=1, random_state=1), "user_id")
    assert set(valid1.user_id) == set(valid2.user_id)
    assert not set(fit1.user_id) & set(valid1.user_id)
    assert len(fit1) + len(valid1) == len(frame)
    assert 0.17 < len(valid1) / len(frame) < 0.23


def test_training_beats_the_mean_baseline_and_reports_rmse(tmp_path, monkeypatch):
    reported = []
    monkeypatch.setattr(task, "report", lambda rmse, step: reported.append((rmse, step)))
    monkeypatch.setattr(task, "load", lambda args: synthetic())
    metrics = task.main(
        [
            "--project=p",
            "--train_table=p.d.t",
            f"--features={','.join(FEATURES)}",
            f"--model_dir={tmp_path / 'trial' / 'model'}",
            "--n_estimators=60",
        ]
    )
    assert metrics["rmse"] < metrics["baseline_rmse"] / 2
    assert metrics["r2"] > 0.8
    assert reported == [(metrics["rmse"], 60)]
    assert metrics["n_train"] + metrics["n_validation"] == 600


def test_saved_model_is_what_the_prebuilt_container_loads(tmp_path):
    frame = synthetic()
    args = args_for(tmp_path)
    booster, metrics, predictions = task.train(frame, args)
    eval_dir = task.save(booster, metrics, predictions, FEATURES, args.model_dir)
    assert sorted(os.listdir(args.model_dir)) == ["model.bst"]
    loaded = xgb.Booster()
    loaded.load_model(os.path.join(args.model_dir, "model.bst"))
    rows = frame[FEATURES].head(5).to_numpy(dtype=float)
    assert np.allclose(loaded.predict(xgb.DMatrix(rows)), booster.predict(xgb.DMatrix(rows)))
    assert os.path.basename(eval_dir) == "eval"
    assert json.load(open(os.path.join(eval_dir, "feature_names.json"))) == FEATURES
    assert json.load(open(os.path.join(eval_dir, "metrics.json")))["rmse"] == metrics["rmse"]


def test_validation_predictions_carry_ids_actuals_and_predictions(tmp_path):
    frame = synthetic()
    args = args_for(tmp_path)
    booster, metrics, predictions = task.train(frame, args)
    eval_dir = task.save(booster, metrics, predictions, FEATURES, args.model_dir)
    saved = pd.read_csv(os.path.join(eval_dir, "validation_predictions.csv"))
    assert list(saved.columns) == ["user_id", "actual", "predicted"]
    assert len(saved) == metrics["n_validation"]
    check = regression_metrics(saved.actual, saved.predicted)
    assert check["rmse"] == pytest.approx(metrics["rmse"], rel=1e-6)
    assert check["mae"] == pytest.approx(metrics["mae"], rel=1e-6)


def test_hyperparameters_are_recorded_and_change_the_model(tmp_path):
    frame = synthetic()
    _, shallow, _ = task.train(frame, args_for(tmp_path, max_depth=1))
    _, deep, _ = task.train(frame, args_for(tmp_path, max_depth=6, learning_rate=0.3))
    assert shallow["hyperparameters"]["max_depth"] == 1 and deep["hyperparameters"]["max_depth"] == 6
    assert shallow["rmse"] != deep["rmse"]


def test_every_tuned_hyperparameter_is_a_trainer_argument():
    tuned = {"max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "min_child_weight"}
    parsed = vars(task.parse_args(["--project=p", "--train_table=t", "--features=a"]))
    assert tuned <= set(parsed)


def test_gcs_paths_use_the_fuse_mount():
    assert task.fuse_path("gs://b/x/model/") == "/gcs/b/x/model/"
    assert task.fuse_path("/tmp/m") == "/tmp/m"


def test_the_staged_package_contains_the_trainer_module(tmp_path):
    import tarfile

    from pipeline.stage_trainer import TRAINER_DIR, build

    package = build(tmp_path)
    with tarfile.open(package) as tar:
        names = tar.getnames()
    assert any(n.endswith("clv_trainer/task.py") for n in names)
    assert not list(TRAINER_DIR.glob("*.egg-info"))
