"""MLP experiment for tabular-playground-series-may-2022.

Same data loading and featurization as the LightGBM baseline (numeric columns
as-is plus the 10 per-character ordinal columns from f_27), but the model is a
standardized-input feed-forward neural network: an sklearn Pipeline of
StandardScaler -> MLPClassifier, fit once on all training rows.

Motivation: the signal in this competition lives in multiplicative interactions
among the continuous manufacturing features. Gradient-boosted trees approximate
smooth multiplicative structure with axis-aligned staircase splits and need many
trees per interaction, whereas even a shallow fully connected network learns
exactly these cross-terms directly in its first hidden layer. Standardizing the
inputs makes the optimization well-conditioned.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COMP = "tabular-playground-series-may-2022"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
STR_COL = "f_27"
STR_LEN = 10
SEED = 0
N_JOBS = int(os.environ.get("EV_N_JOBS", "8"))


class _Tracker:
    """Best-effort MLflow tracking: never lets telemetry kill training."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self._cm = None

    def start(self):
        try:
            import mlflow

            self.mlflow = mlflow
            self._cm = mlflow.start_run()
            self._cm.__enter__()
            self.ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] disabled: {exc}")
            self.ok = False
        return self

    def log_params(self, params):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(params)
        except Exception as exc:
            print(f"[mlflow] log_params failed, disabling: {exc}")
            self.ok = False

    def log_metric(self, key, value, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            print(f"[mlflow] log_metric failed, disabling: {exc}")
            self.ok = False

    def finish(self):
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass


def featurize(df):
    """Numeric columns as-is + f_27 split into one ordinal column per character."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    for i in range(STR_LEN):
        feats[f"{STR_COL}_{i}"] = chars.str[i].map(ord).astype(np.int16) - ord("A")
    return feats


def main():
    tracker = _Tracker().start()
    mlp_kwargs = dict(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=2048,
        learning_rate="constant",
        learning_rate_init=1e-3,
        max_iter=40,
        shuffle=True,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        tol=1e-4,
    )
    params = dict(
        model="sklearn_mlp",
        scaler="standard",
        seed=SEED,
        n_jobs=N_JOBS,
        features="numeric + f_27 per-char ordinal",
        **{f"mlp_{k}": v for k, v in mlp_kwargs.items()},
    )
    tracker.log_params(params)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)[x_all.columns]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**mlp_kwargs)),
    ])
    pipe.fit(x_all, y_all)

    mlp = pipe.named_steps["mlp"]
    print(f"n_iter_={mlp.n_iter_}")
    tracker.log_metric("n_iter", mlp.n_iter_)
    # Internal validation only: the MLP's own early-stopping split on the
    # training data (validation_scores_ are accuracies on that split).
    for i, v in enumerate(getattr(mlp, "validation_scores_", []) or []):
        tracker.log_metric("val_score", v, step=i + 1)
        if (i + 1) % 5 == 0:
            print(f"epoch {i + 1}: val_score={v:.5f}")
    best_val = getattr(mlp, "best_validation_score_", None)
    if best_val is not None:
        print(f"best_validation_score={best_val:.5f}")
        tracker.log_metric("best_validation_score", best_val)
    for i, v in enumerate(getattr(mlp, "loss_curve_", []) or []):
        if (i + 1) % 5 == 0:
            tracker.log_metric("train_loss", v, step=i + 1)

    proba = pipe.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame({"id": test["id"].values, "target": proba})
    # keep sample_submission's exact id order/dtype
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert sub["target"].notna().all(), "missing predictions for some test ids"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading
    tracker.finish()


if __name__ == "__main__":
    main()
