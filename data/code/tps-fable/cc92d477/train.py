"""Baseline for tabular-playground-series-may-2022.

Gradient-boosted trees (LightGBM) on the raw numeric columns plus a simple
per-character ordinal expansion of the categorical string column f_27.
This variant uses LightGBM linear trees (piecewise-linear leaves) on standardized float32 inputs.
No engineered interaction features, no unique-character counts, no tuning.
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
from sklearn.preprocessing import StandardScaler

COMP = "tabular-playground-series-may-2022"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
STR_COL = "f_27"
STR_LEN = 10
SEED = 0
N_JOBS = int(os.environ.get("EV_N_JOBS", "8"))


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


TE_M = 20.0  # smoothing strength of the target encoding


def featurize(df):
    """Numeric columns as-is; f_27 is handled separately via target encoding."""
    return df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])


def smoothed_te(ref_s, ref_y, query_s, m, prior):
    """te(s) = (k_s + m*prior) / (n_s + m) from reference data; unseen -> prior."""
    stats = pd.DataFrame({"s": np.asarray(ref_s), "y": np.asarray(ref_y, dtype=np.float64)})
    grp = stats.groupby("s")["y"].agg(["sum", "count"])
    te = (grp["sum"] + m * prior) / (grp["count"] + m)
    out = pd.Series(np.asarray(query_s)).map(te).fillna(prior).values
    return out.astype(np.float64)


def target_encode_f27(s_tr, y_tr, s_val, s_test, m=TE_M, seed=SEED):
    """Out-of-fold TE for the training split; full-train-split TE for val/test."""
    prior = float(np.mean(y_tr))
    te_tr = np.empty(len(s_tr), dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in skf.split(np.zeros(len(s_tr)), y_tr):
        te_tr[enc_idx] = smoothed_te(s_tr[fit_idx], y_tr[fit_idx], s_tr[enc_idx], m, prior)
    te_val = smoothed_te(s_tr, y_tr, s_val, m, prior)
    te_test = smoothed_te(s_tr, y_tr, s_test, m, prior)
    return te_tr, te_val, te_test


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
        features="numeric + f_27 smoothed target encoding (standardized, float32)",
        te_m=TE_M,
        te_oof_folds=5,
        linear_tree=True,
        linear_lambda=0.01,
    )
    tracker.log_params(params)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    s_all = train[STR_COL].astype(str).values
    x_test = featurize(test)[x_all.columns]
    s_test = test[STR_COL].astype(str).values

    idx_all = np.arange(len(x_all))
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=0.1, stratify=y_all, random_state=SEED
    )
    x_tr, x_val = x_all.iloc[idx_tr].reset_index(drop=True), x_all.iloc[idx_val].reset_index(drop=True)
    y_tr, y_val = y_all[idx_tr], y_all[idx_val]
    s_tr, s_val = s_all[idx_tr], s_all[idx_val]

    # Whole f_27 string as one high-cardinality categorical, summarised by its
    # smoothed target rate; training rows are encoded out-of-fold, val/test rows
    # from the full training split (so val stays an honest holdout).
    te_tr, te_val, te_test = target_encode_f27(s_tr, y_tr, s_val, s_test)
    x_tr = x_tr.copy(); x_val = x_val.copy(); x_test = x_test.copy()
    x_tr[f"{STR_COL}_te"] = te_tr
    x_val[f"{STR_COL}_te"] = te_val
    x_test[f"{STR_COL}_te"] = te_test
    x_test = x_test[x_tr.columns]
    n_distinct = int(pd.Series(s_tr).nunique())
    print(f"f_27 distinct strings in training split: {n_distinct}; n_features={x_tr.shape[1]}")
    tracker.log_params({"f27_n_distinct_train": n_distinct, "n_features": int(x_tr.shape[1])})
    # Linear leaves fit least-squares models per leaf: standardize every column
    # (fit on train only) and use float32 so there are no integer dtypes/NaNs.
    # Split thresholds are scale-invariant, so tree structure is unaffected.
    cols = list(x_tr.columns)
    scaler = StandardScaler().fit(x_tr.values.astype(np.float64))
    x_tr = pd.DataFrame(scaler.transform(x_tr.values).astype(np.float32), columns=cols)
    x_val = pd.DataFrame(scaler.transform(x_val.values).astype(np.float32), columns=cols)
    x_test = pd.DataFrame(scaler.transform(x_test.values).astype(np.float32), columns=cols)
    assert not (np.isnan(x_tr.values).any() or np.isnan(x_test.values).any())

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
        linear_tree=True,
        linear_lambda=0.01,
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
