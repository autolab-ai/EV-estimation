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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                           min_df=2, sublinear_tf=True, max_features=None)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = train["author"].values

    # quick local estimate on a 10% holdout
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    print(f"local_holdout_logloss={log_loss(y_val, clf.predict_proba(x_val)):.5f}")

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
