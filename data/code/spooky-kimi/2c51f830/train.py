"""Baseline for spooky-author-identification.

A solid mid-competition solution: word + character TF-IDF features stacked,
logistic regression with moderate regularization. Targets the historical
leaderboard median (~0.42 log loss); deliberately NOT state of the art.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.

Parent experiment: the regularization strength C is chosen by cross-validation
(mean 5-fold stratified log loss over a grid); the fold-fits are parallelized
with joblib across the Slurm job's cores.

This experiment: the final model is a fixed two-model convex probability blend
of the logistic regression (C from CV) and a ComplementNB(alpha=0.1) fitted on
the identical TF-IDF matrix. The blend weight w is chosen by out-of-fold log
loss over the same 5 stratified folds (w swept over 0.0..1.0 in steps of 0.1).
"""

import os
import re
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0
N_JOBS = int(os.environ.get("N_JOBS", "4"))  # cores in the eval's Slurm job
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CV_FOLDS = 5

# --- best-effort MLflow telemetry: never let tracking kill the run ---------
_mlflow = None
try:
    import mlflow

    _mlflow = mlflow
except Exception as exc:  # pragma: no cover
    print(f"mlflow unavailable, continuing untracked: {exc}")

_mlflow_ok = _mlflow is not None


def log_metric(name, value, step=None):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        _mlflow.log_metric(name, float(value), step=step)
    except Exception as exc:
        _mlflow_ok = False
        print(f"mlflow logging disabled after error: {exc}")


def log_params(params):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        _mlflow.log_params(params)
    except Exception as exc:
        _mlflow_ok = False
        print(f"mlflow logging disabled after error: {exc}")


class _NoOpRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def start_run():
    global _mlflow_ok
    if _mlflow_ok:
        try:
            return _mlflow.start_run()
        except Exception as exc:
            _mlflow_ok = False
            print(f"mlflow start_run failed, continuing untracked: {exc}")
    return _NoOpRun()


# ---------------------------------------------------------------------------


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def _fit_score_fold(x, y, tr_idx, va_idx, C):
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(x[tr_idx], y[tr_idx])
    return log_loss(y[va_idx], clf.predict_proba(x[va_idx]))


def select_C(x_all, y_all):
    """Mean stratified k-fold log loss for each C in C_GRID; return best C."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(x_all, y_all))
    tasks = [(C, fold_i, tr, va)
             for C in C_GRID
             for fold_i, (tr, va) in enumerate(folds)]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_score_fold)(x_all, y_all, tr, va, C)
        for C, fold_i, tr, va in tasks
    )
    losses = {C: [] for C in C_GRID}
    for (C, fold_i, _, _), loss in zip(tasks, results):
        losses[C].append(loss)
    best_C, best_mean = None, np.inf
    for i, C in enumerate(C_GRID):
        mean_loss = float(np.mean(losses[C]))
        print(f"cv_logloss C={C}: {mean_loss:.5f}")
        log_metric("cv_logloss", mean_loss, step=i)
        if mean_loss < best_mean:
            best_C, best_mean = C, mean_loss
    print(f"selected_C={best_C} (cv_logloss={best_mean:.5f})")
    return best_C


def _align(proba, classes):
    """Reorder a predict_proba matrix onto the canonical CLASSES column order."""
    idx = [list(classes).index(c) for c in CLASSES]
    return proba[:, idx]


def _fit_predict_fold(x, y, tr_idx, va_idx, C):
    """Fit LR and ComplementNB on the fold's train part; return aligned val probas."""
    lr = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    lr.fit(x[tr_idx], y[tr_idx])
    p_lr = _align(lr.predict_proba(x[va_idx]), lr.classes_)
    nb = ComplementNB(alpha=0.1, norm=False)
    nb.fit(x[tr_idx], y[tr_idx])
    p_nb = _align(nb.predict_proba(x[va_idx]), nb.classes_)
    return p_lr, p_nb


def select_blend_w(x_all, y_all, best_C):
    """Pick the LR/NB convex blend weight by out-of-fold log loss.

    Uses the same StratifiedKFold splits as select_C: each fold contributes
    validation probabilities from both models (10 fits, parallelized), then
    w is swept over {0.0, 0.1, ..., 1.0} on the collected OOF matrices.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(x_all, y_all))
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_predict_fold)(x_all, y_all, tr, va, best_C)
        for tr, va in folds
    )
    p_lr = np.zeros((len(y_all), len(CLASSES)), dtype=np.float64)
    p_nb = np.zeros((len(y_all), len(CLASSES)), dtype=np.float64)
    for (_, va), (pl, pn) in zip(folds, results):
        p_lr[va] = pl
        p_nb[va] = pn
    best_w, best_loss = 0.0, np.inf
    for i in range(11):
        w = i / 10.0
        p = w * p_lr + (1.0 - w) * p_nb
        p /= p.sum(axis=1, keepdims=True)
        loss = log_loss(y_all, p, labels=CLASSES)
        print(f"oof_blend_logloss w={w:.1f}: {loss:.5f}")
        log_metric("oof_blend_logloss", loss, step=i)
        if loss < best_loss:
            best_w, best_loss = w, loss
    print(f"selected_w={best_w} (oof_blend_logloss={best_loss:.5f})")
    return best_w


def blend_predict(x_tr, y_tr, x_te, C, w):
    """Fit LR + ComplementNB on (x_tr, y_tr) and return blended probas on x_te."""
    lr = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    lr.fit(x_tr, y_tr)
    p_lr = _align(lr.predict_proba(x_te), lr.classes_)
    nb = ComplementNB(alpha=0.1, norm=False)
    nb.fit(x_tr, y_tr)
    p_nb = _align(nb.predict_proba(x_te), nb.classes_)
    proba = (w * p_lr + (1.0 - w) * p_nb).astype(np.float64)
    proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = np.asarray(train["author"].tolist())  # plain ndarray: CV fold indexing needs it

    with start_run():
        log_params({"model": "logreg+complementnb_blend_tfidf_word12_charwb25",
                    "cv_folds": CV_FOLDS, "c_grid": ",".join(map(str, C_GRID)),
                    "nb_alpha": 0.1, "max_iter": 2000,
                    "n_jobs": N_JOBS, "seed": SEED})

        # choose C by cross-validation (was hard-coded C=1.0 in the parent)
        best_C = select_C(x_all, y_all)
        log_metric("selected_C", best_C)

        # choose the LR/ComplementNB blend weight by out-of-fold log loss
        w = select_blend_w(x_all, y_all, best_C)
        log_metric("blend_w", w)

        # quick local estimate of the blend on a 10% holdout
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )
        holdout = log_loss(y_val, blend_predict(x_tr, y_tr, x_val, best_C, w),
                           labels=CLASSES)
        print(f"local_holdout_logloss={holdout:.5f}")
        log_metric("local_holdout_logloss", holdout)

        # refit both models on everything for the real submission
        proba = blend_predict(x_all, y_all, x_test, best_C, w)  # rows sum to 1

        sub = pd.DataFrame(proba, columns=CLASSES)
        sub.insert(0, "id", test["id"])
        sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
        os.makedirs('submission', exist_ok=True)  # the hill grades this directory
        sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
