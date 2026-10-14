"""Compact MLP on the leaf-classification descriptors (EV Bench level-1 node).

Same pipeline as the root baseline (standardised 192 precomputed
shape/margin/texture descriptors plus OOF-selected uniform-shrinkage
calibration), with the linear multinomial logistic regression replaced by a
small early-stopped multilayer perceptron that can model non-linear
interactions between the shape, margin and texture descriptor families.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- best-effort MLflow telemetry: a tracking failure must never break training ----
try:
    import mlflow
    _MLFLOW = True
except Exception as _e:  # pragma: no cover
    print("mlflow unavailable, continuing untracked: %s" % _e)
    _MLFLOW = False


class _NoopRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _start_run():
    global _MLFLOW
    if not _MLFLOW:
        return _NoopRun()
    try:
        return mlflow.start_run()
    except Exception as e:
        print("mlflow start_run failed, continuing untracked: %s" % e)
        _MLFLOW = False
        return _NoopRun()


def _log_metric(name, value, step=None):
    global _MLFLOW
    if not _MLFLOW:
        return
    try:
        mlflow.log_metric(name, float(value), step=step)
    except Exception as e:
        print("mlflow log_metric failed, disabling tracking: %s" % e)
        _MLFLOW = False


def _log_params(params):
    global _MLFLOW
    if not _MLFLOW:
        return
    try:
        mlflow.log_params(params)
    except Exception as e:
        print("mlflow log_params failed, disabling tracking: %s" % e)
        _MLFLOW = False
# ------------------------------------------------------------------------------------

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,               # L2 weight decay
    batch_size="auto",
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    tol=1e-4,
    random_state=SEED,
)


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    # plain numpy strings: with pyarrow installed pandas 3 returns Arrow-backed
    # string arrays here, which scikit-learn splitters cannot fancy-index
    x_all, y_all = featurize(train), train["species"].to_numpy(dtype=str)
    x_test = featurize(test)

    scaler = StandardScaler().fit(x_all)
    x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

    with _start_run():
        _log_params({"model": "MLPClassifier", **{"mlp_%s" % k: v for k, v in MLP_PARAMS.items()}})

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all_s, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
        clf = MLPClassifier(**MLP_PARAMS)
        clf.fit(x_tr, y_tr)
        holdout_loss = log_loss(y_val, clf.predict_proba(x_val), labels=clf.classes_)
        print("local_holdout_logloss=%.5f" % holdout_loss)
        _log_metric("local_holdout_logloss", holdout_loss)

        # Select shrinkage strength alpha (toward the uniform prior over species) by
        # out-of-fold validation on the raw features; the pipeline re-scales per fold.
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(**MLP_PARAMS)),
        ])
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        oof = cross_val_predict(pipe, x_all, y_all, cv=cv, method="predict_proba")
        n_classes = len(classes)
        alphas = np.linspace(0.0, 0.5, 51)
        oof_losses = [log_loss(y_all, (1.0 - a) * oof + a / n_classes, labels=classes) for a in alphas]
        best_idx = int(np.argmin(oof_losses))
        alpha = float(alphas[best_idx])
        print("oof_alpha=%.4f oof_logloss=%.5f" % (alpha, oof_losses[best_idx]))
        _log_metric("oof_alpha", alpha)
        _log_metric("oof_logloss", oof_losses[best_idx])

        clf.fit(x_all_s, y_all)
        proba = clf.predict_proba(x_test_s).astype(np.float64)
        proba = (1.0 - alpha) * proba + alpha / n_classes
        proba = np.clip(proba, 1e-15, 1.0)
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
    main()
