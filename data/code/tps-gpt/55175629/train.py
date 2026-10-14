"""Bounded CPU LightGBM on numeric features and the ten f_27 character ordinals.

The training-only holdout selects a checkpoint; only the frozen evaluator
scores the submission. No additional features, ensembling, or full-data refit.
"""

import time

SCRIPT_START = time.monotonic()

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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


class _DeadlineCallback:
    """Keep the best validation iteration if the absolute deadline is reached."""

    order = 20  # Run before LightGBM's early-stopping callback (order 30).
    before_iteration = False

    def __init__(self, deadline, tracker, learning_rate):
        self.deadline = deadline
        self.tracker = tracker
        self.learning_rate = learning_rate
        self.best_auc = -float("inf")
        self.best_iteration = -1
        self.best_results = None
        self.timed_out = False

    def __call__(self, env):
        results = env.evaluation_result_list
        val_auc = next(
            result[2] for result in results
            if result[0] == "valid_0" and result[1] == "auc"
        )
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_iteration = env.iteration
            self.best_results = list(results)

        metrics = {
            "val_auc": val_auc,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": time.monotonic() - SCRIPT_START,
        }
        for key, value in metrics.items():
            self.tracker.log_metric(key, value, step=env.iteration + 1)
        # Holdout AUC is used internally for selection, not printed as a score.
        print(f"iteration={env.iteration + 1} "
              f"elapsed_seconds={metrics['elapsed_seconds']:.3f}", flush=True)
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            raise lgb.callback.EarlyStopException(
                self.best_iteration, self.best_results
            )


def fit_model(model, x_tr, y_tr, x_val, y_val, config, tracker, deadline):
    deadline_callback = _DeadlineCallback(
        deadline, tracker, config["learning_rate"]
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        callbacks=[
            deadline_callback,
            lgb.early_stopping(
                stopping_rounds=50, first_metric_only=True, verbose=False
            ),
        ],
    )
    tracker.log_metric("best_iteration", model.best_iteration_)
    tracker.log_metric("local_holdout_auc", deadline_callback.best_auc)
    stop_reason = "time limit" if deadline_callback.timed_out else "boosting complete"
    print(f"stop={stop_reason}; best_iteration={model.best_iteration_}", flush=True)


def main():
    config = load_config()
    deadline = SCRIPT_START + config["time_limit_seconds"]
    np.random.seed(SEED)

    with _Tracker() as tracker:
        tracker.log_params(
            dict(
                model="cpu_lightgbm",
                objective="binary",
                metric="auc",
                early_stopping_patience=50,
                seed=SEED,
                holdout_frac=0.1,
                features="numeric + f_27 per-char ordinal",
                scaler="StandardScaler fitted on training partition",
                **config,
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
        y_tr = np.ascontiguousarray(y_tr, dtype=np.float32)

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            num_leaves=config["num_leaves"],
            max_depth=config["max_depth"],
            max_bin=config["max_bin"],
            min_child_samples=config["min_child_samples"],
            n_jobs=config["n_jobs"],
            random_state=SEED,
            verbosity=-1,
        )
        fit_model(model, x_tr, y_tr, x_val, y_val, config, tracker, deadline)
        proba = model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1]
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
