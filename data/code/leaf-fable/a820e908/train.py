"""Leaf-classification: standardised 192 precomputed descriptors + multinomial logistic
regression, with the regularisation strength C chosen by 5-fold stratified CV
(out-of-fold multi-class log loss) instead of the fixed C=1.0 of the parent.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C_GRID = [1, 3, 10, 30, 100, 300, 1000, 2000, 3000, 5000, 10000]
N_SPLITS = 5
N_JOBS = int(os.environ.get("N_JOBS", "4"))


# ---- best-effort MLflow tracking (never allowed to kill the run) -------------
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

    def start(self):
        if not self.ok:
            return None
        try:
            return self.mlflow.start_run()
        except Exception as e:
            print("mlflow start_run failed, continuing untracked: %r" % (e,))
            self.ok = False
            return None

    def params(self, d):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:
            print("mlflow log_params failed, disabling tracking: %r" % (e,))
            self.ok = False

    def metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print("mlflow log_metric failed, disabling tracking: %r" % (e,))
            self.ok = False

    def end(self):
        if not self.ok:
            return
        try:
            self.mlflow.end_run()
        except Exception:
            pass


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def make_clf(C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=10000, tol=1e-6, random_state=SEED)


def fit_fold(C, tr_idx, va_idx, x_all, y_all):
    scaler = StandardScaler().fit(x_all[tr_idx])
    clf = make_clf(C).fit(scaler.transform(x_all[tr_idx]), y_all[tr_idx])
    return C, va_idx, clf.predict_proba(scaler.transform(x_all[va_idx])), list(clf.classes_)


def main():
    trk = _Tracker()
    trk.start()

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all, y_all = featurize(train), train["species"].to_numpy().astype(str)
    x_test = featurize(test)
    sorted_classes = sorted(np.unique(y_all))

    trk.params({"model": "logreg_lbfgs", "n_splits": N_SPLITS, "c_grid": str(C_GRID),
                "max_iter": 10000, "tol": 1e-6, "seed": SEED})

    # Reference-only 80/20 holdout at the parent's C=1 (does not influence the choice of C).
    scaler0 = StandardScaler().fit(x_all)
    x_tr, x_val, y_tr, y_val = train_test_split(
        scaler0.transform(x_all), y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    clf0 = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED).fit(x_tr, y_tr)
    hold = log_loss(y_val, clf0.predict_proba(x_val), labels=clf0.classes_)
    print("local_holdout_logloss=%.5f (C=1, reference only)" % hold)
    trk.metric("local_holdout_logloss", hold)

    # 5-fold stratified CV over the C grid, out-of-fold log loss.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(x_all, y_all))
    jobs = [(C, tr, va) for C in C_GRID for tr, va in folds]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(fit_fold)(C, tr, va, x_all, y_all) for C, tr, va in jobs)

    oof = {C: np.zeros((len(y_all), len(sorted_classes))) for C in C_GRID}
    for C, va_idx, proba, fold_classes in results:
        assert fold_classes == sorted_classes, "class order differs between folds"
        oof[C][va_idx] = proba

    print("%10s %12s" % ("C", "oof_logloss"))
    best_C, best_loss = None, np.inf
    for i, C in enumerate(C_GRID):
        loss = log_loss(y_all, oof[C], labels=sorted_classes)
        print("%10g %12.5f" % (C, loss))
        trk.metric("cv_oof_logloss", loss, step=i)
        trk.metric("cv_C", C, step=i)
        if loss < best_loss:  # strict '<' -> ties go to the smaller C
            best_C, best_loss = C, loss
    print("selected_C=%g selected_oof_logloss=%.5f" % (best_C, best_loss))
    trk.metric("selected_C", best_C)
    trk.metric("selected_oof_logloss", best_loss)

    # Refit on all rows with the selected C and predict the test set.
    scaler = StandardScaler().fit(x_all)
    clf = make_clf(best_C).fit(scaler.transform(x_all), y_all)
    proba = clf.predict_proba(scaler.transform(x_test)).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)
    out = pd.DataFrame(proba, columns=list(clf.classes_))
    for c in classes:
        if c not in out.columns:
            out[c] = 0.0
    sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    trk.end()


if __name__ == "__main__":
    main()
