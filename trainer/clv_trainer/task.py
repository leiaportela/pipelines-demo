"""Train one XGBoost CLV model for a tuning trial and write its model, metrics and validation predictions."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from clv_trainer.metrics import regression_metrics

VALIDATION_MODULUS = 5
MODEL_FILE = "model.bst"


def is_validation(user_ids) -> np.ndarray:
    """Reproducible split: a customer is in validation when a multiplicative hash of the id falls in bucket 0."""
    ids = np.asarray(user_ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761)) % np.uint64(2**32)) % np.uint64(VALIDATION_MODULUS) == 0


def split(frame: pd.DataFrame, id_column: str):
    mask = is_validation(frame[id_column].to_numpy())
    return frame[~mask], frame[mask]


def fuse_path(path: str) -> str:
    """Cloud Storage paths are mounted under /gcs in Vertex custom training."""
    return "/gcs/" + path[len("gs://") :] if path.startswith("gs://") else path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--train_table", required=True, help="project.dataset.table with id, target and features")
    p.add_argument("--features", required=True, help="comma separated, in the order the model expects")
    p.add_argument("--target", default="ltv_180d")
    p.add_argument("--id_column", default="user_id")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dir", default=os.environ.get("AIP_MODEL_DIR", ""))
    return p.parse_args(argv)


def load(args) -> pd.DataFrame:
    from google.cloud import bigquery

    columns = ", ".join([args.id_column, args.target, *args.features.split(",")])
    return bigquery.Client(project=args.project).query(f"SELECT {columns} FROM `{args.train_table}`").to_dataframe()


def train(frame: pd.DataFrame, args):
    features = args.features.split(",")
    fit, valid = split(frame, args.id_column)
    dtrain = xgb.DMatrix(fit[features].to_numpy(dtype=float), label=fit[args.target].to_numpy(dtype=float))
    dvalid = xgb.DMatrix(valid[features].to_numpy(dtype=float), label=valid[args.target].to_numpy(dtype=float))
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "seed": args.seed,
        "nthread": 4,
    }
    booster = xgb.train(params, dtrain, num_boost_round=args.n_estimators)
    predicted = booster.predict(dvalid)
    actual = valid[args.target].to_numpy(dtype=float)
    metrics = regression_metrics(actual, predicted)
    metrics["baseline_rmse"] = regression_metrics(actual, np.full_like(actual, fit[args.target].mean()))["rmse"]
    metrics["n_train"] = int(len(fit))
    metrics["n_validation"] = int(len(valid))
    metrics["hyperparameters"] = {
        k: getattr(args, k)
        for k in ("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "min_child_weight")
    }
    predictions = pd.DataFrame(
        {args.id_column: valid[args.id_column].to_numpy(), "actual": actual, "predicted": predicted}
    )
    return booster, metrics, predictions


def save(booster, metrics, predictions, features, model_dir: str) -> str:
    """Model goes to model_dir as the prebuilt prediction container expects; the rest goes to a sibling eval dir."""
    model_path = fuse_path(model_dir)
    eval_path = os.path.join(os.path.dirname(model_path.rstrip("/")), "eval")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)
    booster.save_model(os.path.join(model_path, MODEL_FILE))
    with open(os.path.join(eval_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(eval_path, "feature_names.json"), "w") as f:
        json.dump(features, f)
    predictions.to_csv(os.path.join(eval_path, "validation_predictions.csv"), index=False)
    return eval_path


def report(rmse: float, step: int) -> None:
    try:
        import hypertune
    except ImportError:
        return
    hypertune.HyperTune().report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="rmse", metric_value=rmse, global_step=step
    )


def main(argv=None) -> dict:
    args = parse_args(argv)
    if not args.model_dir:
        raise SystemExit("--model_dir or AIP_MODEL_DIR is required")
    booster, metrics, predictions = train(load(args), args)
    save(booster, metrics, predictions, args.features.split(","), args.model_dir)
    report(metrics["rmse"], args.n_estimators)
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    main()
