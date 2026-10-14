"""NB-SVM on a word-only TF-IDF stack for spooky-author-identification.

featurize() keeps only the baseline's word vectorizer (1-2 grams); the
character vectorizer is dropped: the chunked text contains non-sentences,
and lexical word features are a cleaner inductive bias than character
fragments that can encode punctuation and chunking artifacts. The head is
a Naive Bayes / SVM-style hybrid: per-class one-vs-rest log count ratios
weight the TF-IDF columns, then one binary logistic regression
per class (C=4.0) is fit on the ratio-weighted matrix. The three positive-
class probability columns are stacked in [EAP, HPL, MWS] order and each row
is renormalized to sum to 1. Writes submission.csv and submission/submission.csv;
the hill's evaluator grades the latter.
"""

import contextlib
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

try:
    import mlflow
    _mlflow_ok = True
except Exception as _exc:  # tracking is best-effort, never fatal
    print(f"mlflow unavailable, tracking disabled: {_exc}")
    _mlflow_ok = False


def _start_run():
    if not _mlflow_ok:
        return contextlib.nullcontext()
    try:
        return mlflow.start_run()
    except Exception as exc:
        print(f"mlflow.start_run failed, running untracked: {exc}")
        return contextlib.nullcontext()


def _log_params(params):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        mlflow.log_params(params)
    except Exception as exc:
        print(f"mlflow log_params failed, disabling tracking: {exc}")
        _mlflow_ok = False


def _log_metric(name, value, **kwargs):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        mlflow.log_metric(name, value, **kwargs)
    except Exception as exc:
        print(f"mlflow log_metric failed, disabling tracking: {exc}")
        _mlflow_ok = False


COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0
NB_ALPHA = 1.0  # Laplace smoothing for the count ratios
LOGREG_C = 4.0  # regularization of each binary classifier


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = word.fit_transform(train_text).tocsr()
    x_test = word.transform(test_text).tocsr()
    return x_train, x_test


def nb_log_ratios(x, y, classes, alpha=NB_ALPHA):
    """Per-class one-vs-rest Laplace-smoothed log count ratios (length-d each)."""
    ratios = {}
    for c in classes:
        p = np.asarray(x[y == c].sum(axis=0)).ravel() + alpha
        q = np.asarray(x[y != c].sum(axis=0)).ravel() + alpha
        ratios[c] = np.log(p / p.sum()) - np.log(q / q.sum())
    return ratios


def fit_nbsvm(x, y, classes):
    """One binary logistic per class on its NB-ratio-weighted features."""
    ratios = nb_log_ratios(x, y, classes)
    models = {}
    for c in classes:
        clf = LogisticRegression(C=LOGREG_C, max_iter=2000, random_state=SEED)
        clf.fit(x.multiply(ratios[c]), (y == c).astype(int))
        models[c] = (clf, ratios[c])
    return models


def nbsvm_proba(models, x, classes):
    """Stack the positive-class probability columns and renormalize rows to 1."""
    cols = []
    for c in classes:
        clf, r = models[c]
        pos = list(clf.classes_).index(1)
        cols.append(clf.predict_proba(x.multiply(r))[:, pos])
    proba = np.column_stack(cols).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)
    return proba


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = train["author"].to_numpy()

    with _start_run():
        _log_params({
            "model": "nb-svm",
            "nb_alpha": NB_ALPHA,
            "logreg_C": LOGREG_C,
            "word_ngram_range": "1-2",
            "seed": SEED,
        })

        # quick local estimate on a 10% holdout; ratios computed on the training fold only
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )
        models = fit_nbsvm(x_tr, y_tr, CLASSES)
        val_proba = nbsvm_proba(models, x_val, CLASSES)
        holdout = log_loss(y_val, val_proba, labels=CLASSES)
        print(f"local_holdout_logloss={holdout:.5f}")
        _log_metric("local_holdout_logloss", holdout)

        # refit ratios + the three binary classifiers on everything for the real submission
        models = fit_nbsvm(x_all, y_all, CLASSES)
        proba = nbsvm_proba(models, x_test, CLASSES)

        sub = pd.DataFrame(proba, columns=CLASSES)
        sub.insert(0, "id", test["id"])
        sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
        os.makedirs('submission', exist_ok=True)  # the hill grades this directory
        sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
