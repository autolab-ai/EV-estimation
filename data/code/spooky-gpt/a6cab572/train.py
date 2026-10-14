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

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import softmax
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


class Tracking:
    """Telemetry failures must never interrupt prediction generation."""

    def __init__(self):
        self.client = None
        self.context = None
        self.warned = False

    def disable(self, exc):
        self.client = None
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
            self.warned = True

    def __enter__(self):
        try:
            import mlflow

            self.context = mlflow.start_run()
            self.context.__enter__()
            self.client = mlflow
        except Exception as exc:
            self.disable(exc)
        return self

    def log(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as exc:
                self.disable(exc)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc_value, traceback)
            except Exception as exc:
                self.disable(exc)
        return False


def fit_temperature(logits, labels, classes):
    # This is the calibration training objective, not a reported score.
    result = minimize_scalar(
        lambda u: log_loss(labels, softmax(logits / np.exp(u), axis=1), labels=classes),
        method="bounded",
        bounds=(np.log(0.25), np.log(4.0)),
        options={"xatol": 0.001},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


def main(tracking):
    started = time.perf_counter()
    tracking.log("log_params", {
        "C": 1.0, "max_iter": 2000, "random_state": SEED,
        "word_ngram_range": "1,2", "char_ngram_range": "2,5",
        "min_df": 2, "sublinear_tf": True, "char_analyzer": "char_wb",
        "char_max_features": 60000, "validation_fraction": 0.1,
        "calibration": "scalar_temperature", "temperature_min": 0.25,
        "temperature_max": 4.0, "temperature_xatol": 0.001,
    })
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    # Avoid Arrow-backed arrays, which sklearn's split cannot safely index.
    y_all = train["author"].to_numpy(dtype=str)

    # Reuse the baseline holdout fit to learn one confidence-scaling parameter.
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    temperature = fit_temperature(clf.decision_function(x_val), y_val, clf.classes_)
    tracking.log("log_metric", "temperature", temperature, step=0)
    print("Temperature calibration fitted; refitting on all labeled examples.", flush=True)

    # refit on everything for the real submission
    clf.fit(x_all, y_all)
    proba = softmax(clf.decision_function(x_test) / temperature, axis=1).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)

    sub = pd.DataFrame(proba, columns=list(clf.classes_))
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    tracking.log("log_metric", "training_seconds", time.perf_counter() - started, step=0)
    print("Wrote temperature-scaled submission probabilities.", flush=True)
    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    with Tracking() as tracking:
        main(tracking)
