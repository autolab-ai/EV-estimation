"""Classify the 192 leaf descriptors with automatically shrunk linear discriminants.

Writes submission.csv and submission/submission.csv using estimator probabilities
without calibration or postprocessing. Only the official evaluator computes scores.
"""

from contextlib import ExitStack, contextmanager
import os
import time

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0


@contextmanager
def tracking_run():
    """Attach to the supplied MLflow run without making telemetry a dependency."""
    stack = ExitStack()
    enabled = False
    warned = False

    def warn(exc):
        nonlocal enabled, warned
        enabled = False
        if not warned:
            print(f"Warning: MLflow tracking disabled: {exc}", flush=True)
            warned = True

    try:
        import mlflow
        stack.enter_context(mlflow.start_run())
        enabled = True
    except Exception as exc:
        warn(exc)

    def log(method, *args, **kwargs):
        if enabled:
            try:
                getattr(mlflow, method)(*args, **kwargs)
            except Exception as exc:
                warn(exc)

    try:
        yield log
    finally:
        try:
            stack.close()
        except Exception as exc:
            warn(exc)


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def make_classifier():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=None)


def check_probabilities(proba, n_rows, n_classes):
    """Check the output contract without computing a predictive score."""
    if proba.shape != (n_rows, n_classes):
        raise ValueError("Unexpected probability array shape")
    if not np.isfinite(proba).all() or np.any(proba < 0) or np.any(proba > 1):
        raise ValueError("Invalid class probabilities")
    if not np.allclose(proba.sum(axis=1), 1.0):
        raise ValueError("Class probabilities do not sum to one")


def main():
    with tracking_run() as log:
        log("log_params", {
            "classifier": "LinearDiscriminantAnalysis",
            "solver": "lsqr",
            "shrinkage": "auto",
            "priors": "empirical",
            "preprocessing": "StandardScaler",
            "validation_fraction": 0.2,
            "random_state": SEED,
            "feature_count": 192,
        })
        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
        sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
        classes = [c for c in sample.columns if c != "id"]

        x_all, y_all = featurize(train), train["species"].to_numpy()
        x_test = featurize(test)

        # Split raw features so the holdout cannot affect standardization.
        x_tr, x_val, y_tr, _ = train_test_split(
            x_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
        started = time.perf_counter()
        scaler = StandardScaler().fit(x_tr)
        clf = make_classifier()
        clf.fit(scaler.transform(x_tr), y_tr)
        val_proba = clf.predict_proba(scaler.transform(x_val))
        check_probabilities(val_proba, len(x_val), len(classes))
        log("log_metric", "validation_fit_seconds", time.perf_counter() - started, step=0)
        print("Holdout probability checks passed; no local score computed.", flush=True)

        # Refit both stages on all labeled rows for the final submission.
        started = time.perf_counter()
        scaler = StandardScaler().fit(x_all)
        clf = make_classifier()
        clf.fit(scaler.transform(x_all), y_all)
        proba = clf.predict_proba(scaler.transform(x_test))
        check_probabilities(proba, len(test), len(classes))
        log("log_metric", "full_fit_seconds", time.perf_counter() - started, step=1)
        out = pd.DataFrame(proba, columns=list(clf.classes_))
        sub = pd.concat([pd.DataFrame({"id": test["id"].to_numpy()}), out[classes]], axis=1)
        sub = sub.loc[:, sample.columns]
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
        print("Wrote both submission files.", flush=True)


if __name__ == "__main__":
    main()
