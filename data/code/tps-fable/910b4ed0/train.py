"""Baseline for tabular-playground-series-may-2022.

Gradient-boosted trees (LightGBM) on the raw numeric columns plus a simple
smoothed out-of-fold target encoding of the full string column f_27 (plus its
unique-character count). No engineered interaction features, no tuning.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

COMP = "tabular-playground-series-may-2022"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
STR_COL = "f_27"
STR_LEN = 10
SEED = 0
N_JOBS = int(os.environ.get("EV_N_JOBS", "8"))
TE_SMOOTH = 20.0  # additive smoothing constant m for the f_27 target encoding
TE_FOLDS = 5


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
    """Numeric columns as-is + f_27 unique-character count (f_27 itself is target-encoded separately)."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    feats[f"{STR_COL}_nunique"] = chars.map(lambda s: len(set(s))).astype(np.int8)
    return feats


def _smoothed_te(fit_keys, fit_y, apply_keys, prior, m):
    """Smoothed per-string target mean: (n*mean + m*prior)/(n+m); unseen strings -> prior."""
    stats = pd.DataFrame({"k": fit_keys.values, "y": fit_y}).groupby("k")["y"].agg(["sum", "count"])
    enc = (stats["sum"] + m * prior) / (stats["count"] + m)
    return apply_keys.map(enc).fillna(prior).astype(np.float32).values


def target_encode_f27(train, test, y, m=TE_SMOOTH, n_splits=TE_FOLDS, seed=SEED):
    """Out-of-fold smoothed target encoding of the full f_27 string for train; full-train fit for test."""
    keys_tr = train[STR_COL].astype(str)
    keys_te = test[STR_COL].astype(str)
    prior = float(np.mean(y))
    te_train = np.full(len(train), prior, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in skf.split(keys_tr, y):
        te_train[enc_idx] = _smoothed_te(
            keys_tr.iloc[fit_idx], y[fit_idx], keys_tr.iloc[enc_idx], prior, m
        )
    te_test = _smoothed_te(keys_tr, y, keys_te, prior, m)
    return te_train, te_test


def main():
    tracker = _Tracker().start()
    params = dict(
        model="lgbm",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        seed=SEED,
        n_jobs=N_JOBS,
        holdout_frac=0.1,
        early_stopping_rounds=100,
        features="numeric + f_27_nunique + f_27 smoothed OOF target encoding",
        te_smoothing=TE_SMOOTH,
        te_folds=TE_FOLDS,
    )
    tracker.log_params(params)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)
    te_train, te_test = target_encode_f27(train, test, y_all)
    x_all[f"{STR_COL}_te"] = te_train
    x_test[f"{STR_COL}_te"] = te_test
    x_test = x_test[x_all.columns]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=N_JOBS,
        random_state=SEED,
    )
    evals_result = {}
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200),
            lgb.record_evaluation(evals_result),
        ],
    )
    for i, v in enumerate(evals_result.get("valid_0", {}).get("auc", [])):
        if (i + 1) % 50 == 0:
            tracker.log_metric("val_auc", v, step=i + 1)
    print(f"best_iteration={model.best_iteration_}")
    tracker.log_metric("best_iteration", model.best_iteration_ or params["n_estimators"])
    val_pred = model.predict_proba(x_val)[:, 1]
    holdout_auc = roc_auc_score(y_val, val_pred)
    print(f"local_holdout_auc={holdout_auc:.5f}")
    tracker.log_metric("local_holdout_auc", holdout_auc)

    proba = model.predict_proba(x_test)[:, 1]
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
