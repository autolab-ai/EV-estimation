"""Word and character TF-IDF with one-vs-rest Naive Bayes weighting.

Writes submission.csv and submission/submission.csv; only the evaluator scores it.
"""

import os
import time

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


class Tracking:
    """Attach to the supplied MLflow run without making telemetry mandatory."""

    def __init__(self):
        self.client = None
        self.run = None
        self.warned = False

    def disable(self, exc):
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
            self.warned = True
        self.client = None

    def __enter__(self):
        try:
            import mlflow
            self.run = mlflow.start_run()
            self.run.__enter__()
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
        if self.run is not None:
            try:
                self.run.__exit__(exc_type, exc_value, traceback)
            except Exception as exc:
                self.disable(exc)
        return False


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def fit_nbsvm(x, y, tracking=None, stage="holdout", step_base=0):
    # Document-presence counts, not TF-IDF sums, determine the NB ratios.
    binary = x.copy().tocsr()
    binary.data.fill(1)
    models = []
    for class_index, author in enumerate(CLASSES):
        started = time.perf_counter()
        positive = y == author
        p = 1 + np.asarray(binary[positive].sum(axis=0)).ravel()
        q = 1 + np.asarray(binary[~positive].sum(axis=0)).ravel()
        ratio = np.log((p / p.sum()) / (q / q.sum()))
        model = LogisticRegression(
            C=4.0, penalty="l2", solver="liblinear", dual=True,
            fit_intercept=True, max_iter=2000, tol=1e-4, random_state=SEED,
        )
        model.fit(x.multiply(ratio).tocsr(), positive)
        models.append((model, ratio))
        elapsed = time.perf_counter() - started
        iterations = int(model.n_iter_.max())
        print(f"{stage}: fitted {author} in {elapsed:.2f}s, "
              f"{iterations} solver iterations", flush=True)
        if tracking is not None:
            step = step_base + class_index
            tracking.log("log_metric", "classifier_fit_seconds", elapsed, step=step)
            tracking.log("log_metric", "solver_iterations", iterations, step=step)
    return models


def predict_nbsvm(models, x):
    columns = []
    for model, ratio in models:
        positive_column = np.flatnonzero(model.classes_ == 1)[0]
        columns.append(model.predict_proba(x.multiply(ratio).tocsr())[:, positive_column])
    proba = np.column_stack(columns).astype(np.float64)
    np.clip(proba, 1e-15, 1 - 1e-15, out=proba)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def main():
    started = time.perf_counter()
    with Tracking() as tracking:
        tracking.log("log_params", {
            "classifier": "one-vs-rest NB-weighted logistic regression",
            "classes": ",".join(CLASSES), "C": 4.0, "penalty": "l2",
            "solver": "liblinear", "dual": True, "fit_intercept": True,
            "max_iter": 2000, "tol": 1e-4, "random_state": SEED,
            "word_ngram_range": "1,2", "char_analyzer": "char_wb",
            "char_ngram_range": "2,5", "min_df": 2,
            "sublinear_tf": True, "char_max_features": 60_000,
            "holdout_fraction": 0.1, "nb_smoothing": 1.0,
            "nb_counting": "binary document presence",
        })
        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
        # Explicit NumPy arrays avoid pandas Arrow-backed split indexing errors.
        train_text = train["text"].to_numpy(dtype=str)
        test_text = test["text"].to_numpy(dtype=str)
        y_all = train["author"].to_numpy(dtype=str)

        feature_started = time.perf_counter()
        x_all, x_test = featurize(train_text, test_text)
        feature_seconds = time.perf_counter() - feature_started
        print(f"Feature extraction completed in {feature_seconds:.2f}s", flush=True)
        tracking.log("log_metric", "feature_seconds", feature_seconds, step=0)

        # Preserve the stratified holdout fit; ratios see only its training rows.
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )
        holdout_models = fit_nbsvm(x_tr, y_tr, tracking, stage="holdout")
        del holdout_models

        # Recompute every ratio and refit every classifier on all labeled rows.
        models = fit_nbsvm(x_all, y_all, tracking, stage="full", step_base=3)
        proba = predict_nbsvm(models, x_test)
        sub = pd.DataFrame(proba, columns=CLASSES)
        sub.insert(0, "id", test["id"])
        sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)
        sub[["id"] + CLASSES].to_csv(
            os.path.join("submission", "submission.csv"), index=False
        )
        total_seconds = time.perf_counter() - started
        print(f"Wrote submission probabilities in {total_seconds:.2f}s", flush=True)
        tracking.log("log_metric", "train_seconds", total_seconds, step=0)


if __name__ == "__main__":
    main()
