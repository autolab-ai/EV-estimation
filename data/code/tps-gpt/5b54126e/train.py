"""Bounded CPU histogram boosting with the baseline's unchanged preprocessing.

The public-training holdout selects an iteration using binary log loss. Only
the frozen evaluator computes AUC; no additional features or full-data refit.
"""

import time

SCRIPT_START = time.monotonic()

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from lightgbm.callback import EarlyStopException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "tabular-playground-series-may-2022"
STR_COL = "f_27"
STR_LEN = 10
SEED = 0


class _Tracker:
    """Best-effort MLflow tracking: never lets telemetry kill training."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self._cm = None

    def __enter__(self):
        try:
            import mlflow

            self.mlflow = mlflow
            self._cm = mlflow.start_run()
            self._cm.__enter__()
            self.ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] disabled: {exc}", flush=True)
            self.ok = False
        return self

    def log_params(self, params):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(params)
        except Exception as exc:
            print(f"[mlflow] log_params failed, disabling: {exc}", flush=True)
            self.ok = False

    def log_metric(self, key, value, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            print(f"[mlflow] log_metric failed, disabling: {exc}", flush=True)
            self.ok = False

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc_value, traceback)
            except Exception:
                pass


def load_config():
    path = os.environ.get("EV_CONFIG", str(Path(__file__).with_name("config.json")))
    with open(path) as handle:
        config = json.load(handle)
    config["data_dir"] = os.environ.get("EV_DATA_DIR", config["data_dir"])
    return config


def featurize(df):
    """Numeric columns as-is + f_27 split into one ordinal column per character."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    for i in range(STR_LEN):
        feats[f"{STR_COL}_{i}"] = chars.str[i].map(ord).astype(np.int16) - ord("A")
    return feats


def build_model(config):
    return LGBMClassifier(**config["model"])


class _IterationCallback:
    """Record the best completed iteration before LightGBM's early stopping."""

    order = 20
    before_iteration = False

    def __init__(self, tracker, deadline, learning_rate):
        self.tracker = tracker
        self.deadline = deadline
        self.learning_rate = learning_rate
        self.best_loss = float("inf")
        self.best_iteration = 0
        self.best_results = None
        self.last_iteration = 0
        self.timed_out = False

    def __call__(self, env):
        iteration = env.iteration + 1
        self.last_iteration = iteration
        # The contract reserves AUC for the evaluator. Use only training loss
        # on the public-training holdout for selecting the model iteration.
        val_loss = next(
            value
            for dataset, metric, value, _ in env.evaluation_result_list
            if dataset == "holdout" and metric == "binary_logloss"
        )
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite holdout loss")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_iteration = iteration
            self.best_results = list(env.evaluation_result_list)

        metrics = {
            "val_loss": val_loss,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": time.monotonic() - SCRIPT_START,
        }
        for key, value in metrics.items():
            self.tracker.log_metric(key, value, step=iteration)
        if iteration == 1 or iteration % 25 == 0:
            print(
                f"iteration={iteration} "
                + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                flush=True,
            )
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            # LightGBM's callback exception uses zero-based iterations.
            raise EarlyStopException(self.best_iteration - 1, self.best_results)


def fit_model(model, x_tr, y_tr, x_val, y_val, config, tracker, deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError("Time budget expired before training")
    progress = _IterationCallback(tracker, deadline, model.learning_rate)
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        eval_names=["holdout"],
        categorical_feature=[],
        callbacks=[
            progress,
            early_stopping(
                stopping_rounds=config["stopping_rounds"],
                first_metric_only=True,
                verbose=False,
            ),
        ],
    )
    if progress.best_iteration == 0:
        raise RuntimeError("No validation checkpoint completed")
    stop_reason = (
        "time limit" if progress.timed_out
        else "early stopping" if progress.last_iteration < model.n_estimators
        else "iteration limit"
    )
    tracker.log_metric("best_iteration", progress.best_iteration)
    print(f"stop={stop_reason}; best_iteration={progress.best_iteration}", flush=True)
    return progress.best_iteration


def main():
    config = load_config()
    deadline = SCRIPT_START + config["time_limit_seconds"]
    np.random.seed(SEED)

    with _Tracker() as tracker:
        tracker.log_params(
            dict(
                model="lightgbm_histogram_gbdt",
                seed=SEED,
                holdout_frac=0.1,
                features="numeric + f_27 per-char ordinal",
                scaler="StandardScaler fitted on training partition",
                time_limit_seconds=config["time_limit_seconds"],
                stopping_rounds=config["stopping_rounds"],
                **config["model"],
            )
        )
        public = Path(config["data_dir"]) / COMP / "prepared" / "public"
        train = pd.read_csv(public / "train.csv")
        test = pd.read_csv(public / "test.csv")
        sample = pd.read_csv(public / "sample_submission.csv")

        x_all = featurize(train)
        y_all = train["target"].values
        x_test = featurize(test)[x_all.columns]
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr.to_numpy(dtype=np.float32))
        x_val = scaler.transform(x_val.to_numpy(dtype=np.float32))
        x_test = scaler.transform(x_test.to_numpy(dtype=np.float32))
        x_tr = np.ascontiguousarray(x_tr, dtype=np.float32)
        x_val = np.ascontiguousarray(x_val, dtype=np.float32)
        x_test = np.ascontiguousarray(x_test, dtype=np.float32)

        model = build_model(config)
        best_iteration = fit_model(
            model, x_tr, y_tr, x_val, y_val, config, tracker, deadline
        )
        proba = model.predict_proba(x_test, num_iteration=best_iteration)[:, 1]
        sub = pd.DataFrame({"id": test["id"].values, "target": proba})
        # keep sample_submission's exact id order/dtype
        sub = sample[["id"]].merge(sub, on="id", how="left")
        assert sub["target"].notna().all(), "missing predictions for some test ids"
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)


if __name__ == "__main__":
    main()
