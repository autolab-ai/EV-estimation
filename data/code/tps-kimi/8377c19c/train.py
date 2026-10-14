"""Baseline for tabular-playground-series-may-2022.

Gradient-boosted trees (LightGBM) on the raw numeric columns, a per-character
categorical expansion of the string column f_27, and order-invariant
bag-of-letters features derived from f_27 (26 per-letter occurrence counts plus
a distinct-character count). No other engineered features, no tuning.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

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


F27_CAT_COLS = [f"{STR_COL}_{i}" for i in range(STR_LEN)]


def featurize(df):
    """Numeric columns as-is + f_27 per-character codes + bag-of-letters counts.

    The ten f_27_* columns hold deterministic 0-25 integer codes (identical for
    train and test) and are meant to be passed to LightGBM as categorical
    features, so it finds many-vs-rest letter partitions instead of threshold
    splits on an arbitrary alphabetical ordering.

    On top of the per-position split, 27 order-invariant numeric features are
    derived from the raw f_27 string: f_27_cnt_<L> counts occurrences of each
    letter A..Z in the 10-character string, and f_27_nunique counts distinct
    characters. These capture which letters appear regardless of position,
    information the per-position categoricals cannot express. They are plain
    numeric features and are NOT passed as categorical.
    """
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    pos_codes = []
    for i, col in enumerate(F27_CAT_COLS):
        code = (chars.str[i].map(ord) - ord("A")).astype(np.int32)
        feats[col] = code
        pos_codes.append(code.to_numpy())
    code_mat = np.stack(pos_codes, axis=1)  # (n_rows, 10) ints in [0, 25]
    n_unique = np.zeros(len(df), dtype=np.int64)
    for k in range(26):
        cnt = (code_mat == k).sum(axis=1)
        feats[f"{STR_COL}_cnt_{chr(ord('A') + k)}"] = cnt.astype(np.int32)
        n_unique += (cnt > 0).astype(np.int64)
    feats[f"{STR_COL}_nunique"] = n_unique.astype(np.int32)
    return feats


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
        features="numeric + f_27 per-char categorical + f_27 letter counts + nunique",
    )
    tracker.log_params(params)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)[x_all.columns]

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
        categorical_feature=F27_CAT_COLS,
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
