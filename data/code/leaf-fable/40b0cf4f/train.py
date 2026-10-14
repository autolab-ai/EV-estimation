"""Leaf-classification: baseline model + post-hoc probability calibration.

Model is unchanged from the parent (StandardScaler + LogisticRegression(C=1.0,
max_iter=5000) on the 192 precomputed descriptors). On top, a two-scalar calibration
layer is fitted on 5-fold out-of-fold predictions:

    Q = P ** a  (row-normalised)      -- power / temperature sharpening
    Q = (1 - eps) * Q + eps / 99       -- uniform floor

(a, eps) are chosen by grid search to minimise the OOF multi-class log loss, then
applied to the test probabilities of the model refit on all training rows.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 1.0
N_SPLITS = 5
A_GRID = np.arange(0.5, 8.01, 0.1)
EPS_GRID = [0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


# ---------------------------------------------------------------------------
# best-effort MLflow tracking (never allowed to kill the run)
class _Tracker:
    def __init__(self):
        self.ok = False
        self.mlflow = None
        try:
            import mlflow  # noqa
            self.mlflow = mlflow
            self.ok = True
        except Exception as e:  # pragma: no cover
            print("mlflow unavailable: %r" % (e,))

    def start_run(self):
        if not self.ok:
            return contextlib.nullcontext()
        try:
            return self.mlflow.start_run()
        except Exception as e:
            print("mlflow start_run failed, continuing untracked: %r" % (e,))
            self.ok = False
            return contextlib.nullcontext()

    def log_params(self, d):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:
            print("mlflow log_params failed, disabling tracking: %r" % (e,))
            self.ok = False

    def log_metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print("mlflow log_metric failed, disabling tracking: %r" % (e,))
            self.ok = False


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def make_model():
    return OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=5000, random_state=SEED), n_jobs=4)


def fit_predict(x_tr, y_tr, x_te):
    scaler = StandardScaler().fit(x_tr)
    clf = make_model().fit(scaler.transform(x_tr), y_tr)
    return clf.predict_proba(scaler.transform(x_te)), list(clf.classes_)


def calib(P, a, eps):
    n_cls = P.shape[1]
    Q = np.clip(P, 1e-12, None) ** a
    Q /= Q.sum(axis=1, keepdims=True)
    return (1.0 - eps) * Q + eps / n_cls


def main():
    tracker = _Tracker()
    with tracker.start_run():
        tracker.log_params({"model": "StandardScaler+OvR(LogisticRegression)", "C": C,
                            "max_iter": 5000, "n_splits": N_SPLITS,
                            "calibration": "power+uniform_floor",
                            "a_grid": "0.5:8.0:0.1", "eps_grid": str(EPS_GRID)})

        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
        sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
        classes = [c for c in sample.columns if c != "id"]

        x_all, y_all = featurize(train), train["species"].values
        x_test = featurize(test)
        sorted_classes = sorted(np.unique(y_all))
        n_cls = len(sorted_classes)

        # (1) out-of-fold predictions
        p_oof = np.zeros((len(y_all), n_cls), dtype=np.float64)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x_all, y_all)):
            p_va, cls = fit_predict(x_all[tr_idx], y_all[tr_idx], x_all[va_idx])
            assert cls == sorted_classes, "fold missing a class"
            p_oof[va_idx] = p_va
            fold_ll = log_loss(y_all[va_idx], p_va, labels=sorted_classes)
            print("fold %d oof_logloss=%.5f" % (fold, fold_ll))
            tracker.log_metric("fold_oof_logloss", fold_ll, step=fold)

        raw_oof = log_loss(y_all, p_oof, labels=sorted_classes)
        print("oof_logloss_uncalibrated=%.5f" % raw_oof)

        # (3) grid search over (a, eps)
        best = (raw_oof, 1.0, 0.0)
        for a in A_GRID:
            for eps in EPS_GRID:
                ll = log_loss(y_all, calib(p_oof, a, eps), labels=sorted_classes)
                if ll < best[0]:
                    best = (ll, float(a), float(eps))
        cal_oof, best_a, best_eps = best
        print("oof_logloss_calibrated=%.5f  a=%.2f eps=%g" % (cal_oof, best_a, best_eps))
        tracker.log_metric("oof_logloss_uncalibrated", raw_oof)
        tracker.log_metric("oof_logloss_calibrated", cal_oof)
        tracker.log_metric("calib_a", best_a)
        tracker.log_metric("calib_eps", best_eps)
        tracker.log_metric("local_holdout_logloss", cal_oof)

        # (4) refit on all rows, predict test, calibrate
        proba, cls = fit_predict(x_all, y_all, x_test)
        proba = calib(proba.astype(np.float64), best_a, best_eps)
        proba = np.clip(proba, 1e-15, 1 - 1e-15)
        proba /= proba.sum(axis=1, keepdims=True)
        out = pd.DataFrame(proba, columns=cls)
        for c in classes:
            if c not in out.columns:
                out[c] = 0.0
        sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
        print("wrote submission/submission.csv rows=%d cols=%d" % sub.shape)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)


if __name__ == "__main__":
    main()
