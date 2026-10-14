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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class _Tracker:
    """Best-effort MLflow logging: disables itself after the first failure."""

    def __init__(self):
        self.ok = mlflow is not None
        self.run = None

    def start(self):
        if not self.ok:
            return
        try:
            self.run = mlflow.start_run()
        except Exception as e:
            print(f"mlflow disabled: {e}")
            self.ok = False

    def log_params(self, d):
        if self.ok:
            try:
                mlflow.log_params(d)
            except Exception as e:
                print(f"mlflow disabled: {e}")
                self.ok = False

    def log_metric(self, k, v, step=None):
        if self.ok:
            try:
                mlflow.log_metric(k, float(v), step=step)
            except Exception as e:
                print(f"mlflow disabled: {e}")
                self.ok = False

    def end(self):
        if self.ok and self.run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

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
    y_all = train["author"].values

    # count features for the naive Bayes model
    cv = CountVectorizer(ngram_range=(1, 2), min_df=2)
    c_all = cv.fit_transform(train["text"])
    c_test = cv.transform(test["text"])

    alphas = [0.03, 0.1, 0.3, 1.0]
    weights = [round(0.1 * i, 1) for i in range(11)]
    tracker = _Tracker()
    tracker.start()
    tracker.log_params({"lr_C": 1.0, "nb_alphas": str(alphas), "n_splits": 5,
                        "count_ngram": "1-2", "count_min_df": 2})

    # 5-fold out-of-fold predictions for LR and for NB at each alpha
    n = len(y_all)
    p_lr = np.zeros((n, 3))
    p_nb = {a: np.zeros((n, 3)) for a in alphas}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(x_all, y_all)):
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        clf.fit(x_all[tr], y_all[tr])
        p_lr[va] = clf.predict_proba(x_all[va])
        for a in alphas:
            nb = MultinomialNB(alpha=a)
            nb.fit(c_all[tr], y_all[tr])
            p_nb[a][va] = nb.predict_proba(c_all[va])
        fold_ll = log_loss(y_all[va], p_lr[va], labels=CLASSES)
        print(f"fold={fold} lr_logloss={fold_ll:.5f}")
        tracker.log_metric("fold_lr_logloss", fold_ll, step=fold)

    lr_oof = log_loss(y_all, p_lr, labels=CLASSES)
    print(f"oof_lr_logloss={lr_oof:.5f}")
    tracker.log_metric("oof_lr_logloss", lr_oof)
    best = (np.inf, None, None)
    for a in alphas:
        nb_oof = log_loss(y_all, p_nb[a], labels=CLASSES)
        print(f"oof_nb_logloss alpha={a} {nb_oof:.5f}")
        for w in weights:
            ll = log_loss(y_all, w * p_lr + (1 - w) * p_nb[a], labels=CLASSES)
            if ll < best[0]:
                best = (ll, a, w)
    best_ll, best_alpha, best_w = best
    print(f"best_alpha={best_alpha} best_w={best_w} local_oof_logloss={best_ll:.5f}")
    tracker.log_params({"best_alpha": best_alpha, "best_w": best_w})
    tracker.log_metric("local_oof_logloss", best_ll)

    # refit on everything for the real submission
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(x_all, y_all)
    nb = MultinomialNB(alpha=best_alpha)
    nb.fit(c_all, y_all)
    assert list(clf.classes_) == CLASSES and list(nb.classes_) == CLASSES
    proba = best_w * clf.predict_proba(x_test) + (1 - best_w) * nb.predict_proba(c_test)
    proba = np.clip(proba.astype(np.float64), 1e-6, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)
    tracker.end()

    sub = pd.DataFrame(proba, columns=list(clf.classes_))
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
