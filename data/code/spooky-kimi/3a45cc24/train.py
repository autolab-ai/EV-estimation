"""Baseline for spooky-author-identification.

A solid mid-competition solution: word + character TF-IDF features stacked,
logistic regression with moderate regularization. Targets the historical
leaderboard median (~0.42 log loss); deliberately NOT state of the art.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.

This experiment: the logistic regression classifier is swapped for
Multinomial Naive Bayes, the canonical model for sparse non-negative text
features, with its smoothing parameter alpha chosen by the same
cross-validation procedure (mean 5-fold stratified log loss over a grid).
The 30 fold-fits are parallelized with joblib across the Slurm job's cores.
"""

import os
import re
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, train_test_split

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0
N_JOBS = int(os.environ.get("N_JOBS", "4"))  # cores in the eval's Slurm job
ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
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


def _fit_score_fold(x, y, tr_idx, va_idx, alpha):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(x[tr_idx], y[tr_idx])
    return log_loss(y[va_idx], clf.predict_proba(x[va_idx]))


def select_alpha(x_all, y_all):
    """Mean stratified k-fold log loss for each alpha in ALPHA_GRID; return best alpha."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(x_all, y_all))
    tasks = [(alpha, fold_i, tr, va)
             for alpha in ALPHA_GRID
             for fold_i, (tr, va) in enumerate(folds)]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_score_fold)(x_all, y_all, tr, va, alpha)
        for alpha, fold_i, tr, va in tasks
    )
    losses = {alpha: [] for alpha in ALPHA_GRID}
    for (alpha, fold_i, _, _), loss in zip(tasks, results):
        losses[alpha].append(loss)
    best_alpha, best_mean = None, np.inf
    for i, alpha in enumerate(ALPHA_GRID):
        mean_loss = float(np.mean(losses[alpha]))
        print(f"cv_logloss alpha={alpha}: {mean_loss:.5f}")
        log_metric("cv_logloss", mean_loss, step=i)
        if mean_loss < best_mean:
            best_alpha, best_mean = alpha, mean_loss
    print(f"selected_alpha={best_alpha} (cv_logloss={best_mean:.5f})")
    return best_alpha


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = np.asarray(train["author"].tolist())  # plain ndarray: CV fold indexing needs it

    with start_run():
        log_params({"model": "multinomial_nb_tfidf_word12_charwb25", "cv_folds": CV_FOLDS,
                    "alpha_grid": ",".join(map(str, ALPHA_GRID)),
                    "n_jobs": N_JOBS, "seed": SEED})

        # choose alpha by cross-validation (same procedure as the parent's C search)
        best_alpha = select_alpha(x_all, y_all)
        log_metric("selected_alpha", best_alpha)

        # quick local estimate on a 10% holdout
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )
        clf = MultinomialNB(alpha=best_alpha)
        clf.fit(x_tr, y_tr)
        holdout = log_loss(y_val, clf.predict_proba(x_val))
        print(f"local_holdout_logloss={holdout:.5f}")
        log_metric("local_holdout_logloss", holdout)

        # refit on everything for the real submission
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
    main()
