"""Baseline for spooky-author-identification.

A solid mid-competition solution: word + character TF-IDF features stacked,
logistic regression with moderate regularization. Targets the historical
leaderboard median (~0.42 log loss); deliberately NOT state of the art.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

_MLFLOW_OK = mlflow is not None


def _log_params(d):
    global _MLFLOW_OK
    if not _MLFLOW_OK:
        return
    try:
        mlflow.log_params(d)
    except Exception as e:  # tracking is best-effort
        print(f"mlflow disabled: {e}")
        _MLFLOW_OK = False


def _log_metric(k, v, step=None):
    global _MLFLOW_OK
    if not _MLFLOW_OK:
        return
    try:
        mlflow.log_metric(k, float(v), step=step)
    except Exception as e:
        print(f"mlflow disabled: {e}")
        _MLFLOW_OK = False

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = np.asarray(train["author"].tolist(), dtype=object)

    # cross-validated choice of regularization strength over Cs, then refit on all rows.
    # (LogisticRegressionCV's warm-started path returned identical scores for every
    # C >= 3 with sklearn 1.9, so the folds are fit explicitly here.)
    Cs = [1, 3, 10, 30, 100]
    _log_params({"Cs": str(Cs), "cv_folds": 5, "solver": "lbfgs", "max_iter": 3000})
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def fold_scores(tr_idx, va_idx):
        out = []
        for c in Cs:
            m = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=SEED)
            m.fit(x_all[tr_idx], y_all[tr_idx])
            out.append(log_loss(y_all[va_idx], m.predict_proba(x_all[va_idx]), labels=m.classes_))
        return out

    scores = np.array(Parallel(n_jobs=5)(
        delayed(fold_scores)(tr, va) for tr, va in skf.split(x_all, y_all)))
    mean_ll = scores.mean(axis=0)
    for c, ll in zip(Cs, mean_ll):
        print(f"cv_logloss C={c:g}: {ll:.5f}")
        _log_metric("cv_logloss", ll, step=int(c))
    best_i = int(mean_ll.argmin())
    best_c = float(Cs[best_i])
    cv_ll = float(mean_ll[best_i])
    print(f"chosen_C={best_c:g}")
    print(f"local_holdout_logloss={cv_ll:.5f}")
    _log_metric("chosen_C", best_c)
    _log_metric("local_holdout_logloss", cv_ll)

    clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=3000, random_state=SEED)
    clf.fit(x_all, y_all)
    proba = clf.predict_proba(x_test).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)

    sub = pd.DataFrame(proba, columns=list(clf.classes_))
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    if mlflow is not None:
        try:
            with mlflow.start_run():
                main()
        except Exception as e:
            if os.path.exists(os.path.join("submission", "submission.csv")):
                print(f"mlflow error after run: {e}")
            else:
                print(f"mlflow start failed ({e}); running untracked")
                _MLFLOW_OK = False
                main()
    else:
        main()
