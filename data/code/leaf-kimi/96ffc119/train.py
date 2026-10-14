"""Leaf-classification: 192 precomputed shape/margin/texture descriptors fed
raw (trees are invariant to monotonic feature transforms, so no scaling) to
sklearn's histogram-based gradient-boosted decision trees, which model the
nonlinear descriptor interactions a linear model can only approximate.
Early stopping on an internal 15% validation split selects the number of
boosting rounds by multiclass log loss, so no outer CV loop is needed.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0

PARAMS = dict(
    loss="log_loss",
    learning_rate=0.06,
    max_iter=1000,
    max_leaf_nodes=31,
    min_samples_leaf=10,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=SEED,
)


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def _mlflow_setup():
    """Best-effort MLflow: attach to the preconfigured run, else a no-op context.

    Tracking must never kill training, so any failure disables it.
    """
    try:
        import mlflow

        return mlflow.start_run(), mlflow
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"mlflow unavailable, continuing untracked: {exc}")
        return contextlib.nullcontext(), None


def _log(mlflow, kind, payload):
    if mlflow is None:
        return
    try:
        if kind == "params":
            mlflow.log_params(payload)
        else:
            for key, value in payload.items():
                mlflow.log_metric(key, float(value))
    except Exception:  # noqa: BLE001
        pass


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all, y_all = featurize(train), train["species"].values
    x_test = featurize(test)

    run_ctx, mlflow = _mlflow_setup()
    with run_ctx:
        _log(mlflow, "params", {"model": "HistGradientBoostingClassifier", **PARAMS})

        clf = HistGradientBoostingClassifier(**PARAMS)
        clf.fit(x_all, y_all)

        n_rounds = int(clf.n_iter_)
        # validation_score_ stores NEGATIVE log loss; the best round is its max.
        best_val_log_loss = float(-np.max(clf.validation_score_)) if clf.validation_score_ is not None else float("nan")
        train_log_loss = float(log_loss(y_all, clf.predict_proba(x_all)))
        print(f"n_boost_rounds={n_rounds}")
        print(f"best_val_log_loss={best_val_log_loss:.5f}")
        print(f"train_log_loss={train_log_loss:.5f}")
        _log(
            mlflow,
            "metrics",
            {
                "n_boost_rounds": n_rounds,
                "best_val_log_loss": best_val_log_loss,
                "train_log_loss": train_log_loss,
            },
        )

        proba = clf.predict_proba(x_test).astype(np.float64)
        proba /= proba.sum(axis=1, keepdims=True)
        out = pd.DataFrame(proba, columns=list(clf.classes_))
        for c in classes:
            if c not in out.columns:
                out[c] = 0.0
        sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
