"""NB-SVM on top of the baseline TF-IDF features for spooky-author-identification.

Same word + character TF-IDF stack as the baseline, but the single multinomial
logistic regression is replaced by a Naive Bayes / SVM hybrid: per-class log
count-ratio vectors r_c (Laplace alpha=1) weight the TF-IDF columns, and one
binary logistic regression per class is fitted on the r_c-weighted matrix.
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
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def nbsvm_proba(x_fit, y_fit, x_score):
    """NB-SVM: per-class log count-ratios weight the columns, then one binary
    logistic regression per class on the weighted matrix. Returns an
    (n_samples, 3) probability array with columns in CLASSES order."""
    cols = []
    for c in CLASSES:
        pos = y_fit == c
        p = np.asarray(x_fit[pos].sum(axis=0)).ravel() + 1.0   # Laplace alpha=1
        q = np.asarray(x_fit[~pos].sum(axis=0)).ravel() + 1.0
        r = np.log(p / p.sum()) - np.log(q / q.sum())
        clf = LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)
        clf.fit(x_fit.multiply(r), pos)
        cols.append(clf.predict_proba(x_score.multiply(r))[:, list(clf.classes_).index(True)])
    proba = np.column_stack(cols).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = train["author"].values

    # quick local estimate on a 10% holdout (r_c from the training fold only)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )
    print(f"local_holdout_logloss={log_loss(y_val, nbsvm_proba(x_tr, y_tr, x_val)):.5f}")

    # refit on everything for the real submission
    proba = nbsvm_proba(x_all, y_all, x_test)  # rows already sum to 1

    sub = pd.DataFrame(proba, columns=CLASSES)
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
