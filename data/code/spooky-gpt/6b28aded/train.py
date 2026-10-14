"""Baseline for spooky-author-identification.

A solid mid-competition solution: word + character TF-IDF features stacked,
logistic regression with moderate regularization. Targets the historical
leaderboard median (~0.42 log loss); deliberately NOT state of the art.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess
import time

from tracking import log_metric, log_params, tracking_run

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    log_params({
        "C": 2.0, "max_iter": 2000, "seed": SEED,
        "holdout_fraction": 0.1, "word_ngram_range": "1,2",
        "char_analyzer": "char_wb", "char_ngram_range": "2,5",
        "min_df": 2, "sublinear_tf": True, "char_max_features": 60_000,
    })
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = train["author"].to_numpy(dtype=str)

    # preserve the stratified holdout fit; scoring belongs to the evaluator
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )
    clf = LogisticRegression(C=2.0, max_iter=2000, random_state=SEED)
    started = time.perf_counter()
    clf.fit(x_tr, y_tr)
    log_metric("fit_seconds", time.perf_counter() - started, step=0)
    log_metric("solver_iterations", int(clf.n_iter_.max()), step=0)

    # refit on everything for the real submission
    started = time.perf_counter()
    clf.fit(x_all, y_all)
    log_metric("fit_seconds", time.perf_counter() - started, step=1)
    log_metric("solver_iterations", int(clf.n_iter_.max()), step=1)
    proba = clf.predict_proba(x_test).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)

    sub = pd.DataFrame(proba, columns=list(clf.classes_))
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    with tracking_run():
        main()
