"""Baseline for tabular-playground-series-may-2022.

LightGBM binary classifier on raw numeric columns plus a simple per-character
expansion of the categorical string column f_27 (the ten letter columns are
declared categorical). No engineered interaction features, no unique-character counts.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

COMP = "tabular-playground-series-may-2022"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
STR_COL = "f_27"
STR_LEN = 10
SEED = 0
N_THREADS = int(os.environ.get("EV_N_THREADS", "4"))
NUM_ROUNDS = int(os.environ.get("EV_NUM_ROUNDS", "2000"))
EARLY_STOP = 100
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=N_THREADS,
    seed=SEED,
    verbose=-1,
)
TRAIN_TIME_LIMIT_S = 9 * 60


class _Tracker:
    """Best-effort MLflow tracking: never lets telemetry kill training."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self._cm = None

    def start(self):
        try:
            import mlflow

            self.mlflow = mlflow
            self._cm = mlflow.start_run()
            self._cm.__enter__()
            self.ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] disabled: {exc}")
            self.ok = False
        return self

    def log_params(self, params):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(params)
        except Exception as exc:
            print(f"[mlflow] log_params failed, disabling: {exc}")
            self.ok = False

    def log_metric(self, key, value, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            print(f"[mlflow] log_metric failed, disabling: {exc}")
            self.ok = False

    def finish(self):
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass


def featurize(df):
    """Numeric columns as-is + f_27 split into one ordinal column per character."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    for i in range(STR_LEN):
        feats[f"{STR_COL}_{i}"] = chars.str[i].map(ord).astype(np.int16) - ord("A")
    return feats


def _time_guard(t_start, limit_s):
    """LightGBM callback: stop boosting once the wall-clock limit is exceeded."""

    def _cb(env):
        if time.time() - t_start > limit_s:
            print(f"time guard hit at iteration {env.iteration + 1}; stopping training")
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    _cb.order = 30
    return _cb


def main():
    tracker = _Tracker().start()
    params = dict(
        model="lightgbm",
        num_boost_round=NUM_ROUNDS,
        early_stopping_rounds=EARLY_STOP,
        seed=SEED,
        num_threads=N_THREADS,
        holdout_frac=0.1,
        features="numeric + f_27 per-char (categorical)",
        **{k: v for k, v in LGB_PARAMS.items() if k not in ("seed", "num_threads", "verbose")},
    )
    tracker.log_params(params)

    np.random.seed(SEED)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)[x_all.columns]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )

    cat_cols = [f"{STR_COL}_{i}" for i in range(STR_LEN)]
    d_tr = lgb.Dataset(x_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    d_val = lgb.Dataset(x_val, label=y_val, categorical_feature=cat_cols, reference=d_tr,
                        free_raw_data=False)

    evals = {}
    t_start = time.time()
    model = lgb.train(
        LGB_PARAMS,
        d_tr,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[d_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=True),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals),
            _time_guard(t_start, TRAIN_TIME_LIMIT_S),
        ],
    )
    print(f"training took {time.time() - t_start:.0f}s")
    for i, auc in enumerate(evals["val"]["auc"]):
        if (i + 1) % 100 == 0 or i + 1 == len(evals["val"]["auc"]):
            tracker.log_metric("val_auc", auc, step=i + 1)

    best_iter = model.best_iteration or model.current_iteration()
    print(f"best_iteration={best_iter}")
    tracker.log_metric("best_iteration", best_iter)
    holdout_auc = roc_auc_score(y_val, model.predict(x_val, num_iteration=best_iter))
    print(f"local_holdout_auc={holdout_auc:.5f}")
    tracker.log_metric("local_holdout_auc", holdout_auc)

    proba = model.predict(x_test, num_iteration=best_iter)
    sub = pd.DataFrame({"id": test["id"].values, "target": proba})
    # keep sample_submission's exact id order/dtype
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert sub["target"].notna().all(), "missing predictions for some test ids"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading
    tracker.finish()


if __name__ == "__main__":
    main()
