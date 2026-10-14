"""Select logistic-regression regularization using fold-local standardization.

Writes submission.csv and submission/submission.csv; the hill grades the latter.
"""

import os
import time

# Set before numerical imports, including in joblib worker processes.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]
SEARCH_JOBS = max(1, min(4, int(os.environ.get("LEAF_SEARCH_JOBS", "4"))))


class Tracking:
    """Tracking failures never interrupt training or submission generation."""

    def __init__(self):
        self.client = None
        self.run = None
        self.warned = False

    def disable(self, error):
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {error}", flush=True)
            self.warned = True
        self.client = None

    def __enter__(self):
        try:
            import mlflow
            self.client = mlflow
            self.run = mlflow.start_run()
            self.run.__enter__()
        except Exception as error:
            self.disable(error)
        return self

    def log(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as error:
                self.disable(error)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            try:
                self.run.__exit__(exc_type, exc_value, traceback)
            except Exception as error:
                self.disable(error)
        return False


def make_search():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs", max_iter=5000, random_state=SEED)),
    ])
    return GridSearchCV(
        pipeline,
        {"classifier__C": C_VALUES},
        scoring="neg_log_loss",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        refit=True,
        n_jobs=SEARCH_JOBS,
        pre_dispatch=SEARCH_JOBS,
        error_score="raise",
    )


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def train_and_predict(tracking):
    tracking.log("log_params", {
        "model": "logistic_regression",
        "solver": "lbfgs",
        "max_iter": 5000,
        "random_state": SEED,
        "c_candidates": C_VALUES,
        "cv_folds": 3,
        "cv_shuffle": True,
        "scoring": "neg_log_loss",
        "refit": True,
        "scaling": "fold_local_standard_scaler",
        "search_jobs": SEARCH_JOBS,
        "blas_threads": 1,
        "features": "all_192_descriptors",
    })
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all, y_all = featurize(train), train["species"].to_numpy(dtype=str)
    x_test = featurize(test)

    search = make_search()
    print("Selecting regularization with three-fold cross-validation.", flush=True)
    started = time.monotonic()
    with threadpool_limits(limits=1), parallel_backend(
        "loky", inner_max_num_threads=1
    ):
        search.fit(x_all, y_all)
        clf = search.best_estimator_
        proba = clf.predict_proba(x_test).astype(np.float64)
    elapsed = time.monotonic() - started
    tracking.log("log_params", {"selected_c": search.best_params_["classifier__C"]})
    tracking.log("log_metric", "fit_seconds", elapsed, step=0)
    print("Cross-validation and refit completed.", flush=True)
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


def main():
    with Tracking() as tracking:
        train_and_predict(tracking)


if __name__ == "__main__":
    main()
