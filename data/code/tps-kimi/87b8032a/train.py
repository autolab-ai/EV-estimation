"""ExtraTrees bagging ensemble for tabular-playground-series-may-2022.

Replaces the two-stage LightGBM boosting procedure with a single
sklearn.ensemble.ExtraTreesClassifier trained once on all training rows.
Extremely randomized trees average many deeply grown, decorrelated trees
with random split thresholds, trading a small amount of bias for a large
variance reduction on this noisy simulated sensor data, and removing the
fragile best_iteration / 1/0.9 refit heuristic entirely.
Same featurization as the baseline: raw numeric columns plus a per-character
ordinal expansion of the string column f_27.
Internal validation is a 5-fold StratifiedKFold cross-validated ROC AUC.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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


def featurize(df):
    """Numeric columns as-is + f_27 split into one ordinal column per character."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    for i in range(STR_LEN):
        feats[f"{STR_COL}_{i}"] = chars.str[i].map(ord).astype(np.int16) - ord("A")
    return feats


def main():
    tracker = _Tracker().start()
    params = dict(
        model="extra_trees",
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        class_weight=None,
        seed=SEED,
        n_jobs=N_JOBS,
        cv_folds=5,
        cv_n_jobs=min(N_JOBS, 2),
        features="numeric + f_27 per-char ordinal",
    )
    tracker.log_params(params)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)[x_all.columns]

    model = ExtraTreesClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        class_weight=None,
        n_jobs=N_JOBS,
        random_state=SEED,
    )

    # Single fit on 100% of the training rows; no holdout, no early stopping.
    model.fit(x_all, y_all)
    proba = model.predict_proba(x_test)[:, 1]

    # Internal validation only: 5-fold stratified cross-validated ROC AUC.
    # One 300-tree fit peaks at ~14GB, so N_JOBS concurrent fold fits would
    # exceed the 64GB eval job; cap fold concurrency at 2 (still n_jobs=N_JOBS
    # threads inside each fit). Guarded: telemetry failure must not kill the run.
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_proba = cross_val_predict(
            model, x_all, y_all, cv=cv, method="predict_proba",
            n_jobs=min(N_JOBS, 2),
        )[:, 1]
        cv_auc = roc_auc_score(y_all, cv_proba)
        print(f"cv_auc={cv_auc:.5f}")
        tracker.log_metric("cv_auc", cv_auc)
    except Exception as exc:
        print(f"[cv] internal validation failed, continuing: {exc}")

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
