"""Word and character TF-IDF with class-specific NB-weighted logistic models.

Writes submission.csv and submission/submission.csv; the evaluator grades the latter.
"""

import os
import re
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from telemetry import tracking

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def main():
    with tracking() as log:
        train_model(log)


def train_model(log):
    log("log_params", {
        "word_ngram_range": "1,2", "char_analyzer": "char",
        "char_ngram_range": "2,5", "min_df": 2, "sublinear_tf": True,
        "char_max_features": 60_000, "C": 1.0, "max_iter": 2000,
        "random_state": SEED, "classifier": "nb_weighted_ovr_logistic",
        "solver": "liblinear", "penalty": "l2", "dual": True,
        "nb_smoothing": 1.0, "nb_features": "binary_occurrence",
    })
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    started = perf_counter()
    x_all, x_test = featurize(train["text"], test["text"])
    log("log_metric", "feature_seconds", perf_counter() - started, step=0)
    y_all = train["author"].values

    # Fit on all public training labels; scoring belongs to the evaluator.
    occurrence = x_all.copy()
    occurrence.data[:] = 1
    probabilities = []
    for class_index, author in enumerate(CLASSES):
        y_class = y_all == author
        a = 1.0 + np.asarray(occurrence[y_class].sum(axis=0)).ravel()
        b = 1.0 + np.asarray(occurrence[~y_class].sum(axis=0)).ravel()
        weights = np.log(a / a.sum()) - np.log(b / b.sum())
        clf = LogisticRegression(
            C=1.0, solver="liblinear", penalty="l2", dual=True,
            max_iter=2000, random_state=SEED,
        )
        started = perf_counter()
        clf.fit(x_all.multiply(weights).tocsr(), y_class.astype(int))
        log("log_metric", "fit_seconds", perf_counter() - started, step=class_index)
        log("log_metric", "solver_iterations", int(clf.n_iter_.max()), step=class_index)
        positive_column = np.flatnonzero(clf.classes_ == 1)[0]
        probabilities.append(
            clf.predict_proba(x_test.multiply(weights).tocsr())[:, positive_column]
        )
    proba = np.column_stack(probabilities).astype(np.float64)
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    proba /= proba.sum(axis=1, keepdims=True)

    sub = pd.DataFrame(proba, columns=CLASSES)
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
