"""Classify leaves with standardized Nyström RBF features and logistic regression.

Writes submission.csv and submission/submission.csv; only the evaluator scores it.
"""

import os
from contextlib import contextmanager
from time import perf_counter

# Limit numerical libraries before importing them, including libraries loaded later.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class Tracking:
    """Best-effort telemetry that never changes or interrupts model training."""

    def __init__(self):
        self.client = None
        self.warned = False

    def disable(self, error):
        if not self.warned:
            print(f"Warning: MLflow tracking disabled: {error}", flush=True)
            self.warned = True
        self.client = None

    def call(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as error:
                self.disable(error)


@contextmanager
def tracking_run():
    tracking = Tracking()
    run = None
    try:
        import mlflow
        run = mlflow.start_run()
        tracking.client = mlflow
    except Exception as error:
        tracking.disable(error)
    try:
        yield tracking
    finally:
        if run is not None:
            try:
                mlflow.end_run()
            except Exception as error:
                tracking.disable(error)

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 1.0


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def make_pipeline():
    return Pipeline([
        ("input_scaler", StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", gamma=1.0 / 192.0,
                              n_components=512, random_state=0)),
        ("feature_scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs",
                                           max_iter=5000, random_state=0)),
    ])


def main(tracking):
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all, y_all = featurize(train), train["species"].to_numpy(dtype=str)
    x_test = featurize(test)
    tracking.call("log_params", {
        "representation": "nystroem_rbf", "gamma": 1.0 / 192.0,
        "n_components": 512, "C": C, "solver": "lbfgs",
        "max_iter": 5000, "random_state": SEED, "holdout_fraction": 0.2,
        "cpu_threads": 4,
    })

    x_tr, x_val, y_tr, _ = train_test_split(
        x_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    clf = make_pipeline()
    start = perf_counter()
    clf.fit(x_tr, y_tr)
    val_proba = clf.predict_proba(x_val)
    assert np.isfinite(val_proba).all()
    assert np.allclose(val_proba.sum(axis=1), 1.0)
    elapsed = perf_counter() - start
    tracking.call("log_metric", "holdout_fit_seconds", elapsed, step=0)
    print(f"Holdout fit and probability checks completed in {elapsed:.2f}s", flush=True)

    start = perf_counter()
    clf.fit(x_all, y_all)
    elapsed = perf_counter() - start
    tracking.call("log_metric", "full_fit_seconds", elapsed, step=1)
    print(f"Full-data fit completed in {elapsed:.2f}s", flush=True)
    proba = clf.predict_proba(x_test).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)
    out = pd.DataFrame(proba, columns=list(clf.classes_))
    for c in classes:
        if c not in out.columns:
            out[c] = 0.0
    sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=4), tracking_run() as tracking:
        main(tracking)
