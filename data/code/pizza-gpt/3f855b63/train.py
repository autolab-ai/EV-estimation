"""Blend the baseline logistic regression with metadata-only CatBoost.

Keep the baseline word TF-IDF and nine transformed requester-history features.
CatBoost uses only the final nine metadata columns; predictions use a fixed
50/50 probability average, with no hyperparameter search or local scoring.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 0.3                      # modest, untuned regularisation
MAX_FEATURES = 5_000         # unigrams only
NUM_COLS = ["requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request"]


def load(name):
    with open(os.path.join(PUBLIC, name)) as f:
        return pd.DataFrame(json.load(f))


def text_of(df):
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("")
    body = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * len(df)))).fillna("")
    return (title + " " + body).astype(str)


def featurize(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    return hstack([xt, csr_matrix(nt)]).tocsr(), hstack([xs, csr_matrix(ns)]).tocsr()


CATBOOST_PARAMS = dict(
    iterations=600, depth=5, learning_rate=0.03, l2_leaf_reg=5,
    loss_function="Logloss", task_type="CPU", thread_count=4,
    random_seed=0, verbose=False, allow_writing_files=False,
)


class Tracking:
    """Attach to the supplied MLflow run; telemetry must never stop training."""

    def __init__(self):
        self.backend = None
        self.run = None
        self.warned = False

    def disable(self, exc):
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
            self.warned = True
        self.backend = None

    def __enter__(self):
        try:
            import mlflow
            self.run = mlflow.start_run()
            self.backend = mlflow
        except Exception as exc:
            self.disable(exc)
        return self

    def log(self, method, *args, **kwargs):
        if self.backend is not None:
            try:
                getattr(self.backend, method)(*args, **kwargs)
            except Exception as exc:
                self.disable(exc)

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            try:
                self.run.__exit__(exc_type, exc, traceback)
            except Exception as tracking_exc:
                self.disable(tracking_exc)
        return False


def main():
    with Tracking() as tracking:
        tracking.log("log_params", {
            "logistic_C": C, "logistic_max_iter": 2000,
            "max_features": MAX_FEATURES, "seed": SEED,
            "catboost_weight": 0.5, "holdout_fraction": 0.2,
            "catboost_features": ",".join(NUM_COLS),
            **{f"catboost_{key}": value for key, value in CATBOOST_PARAMS.items()},
        })
        train, test = load("train.json"), load("test.json")
        y = train["requester_received_pizza"].astype(int).to_numpy()
        x_all, x_test = featurize(train, test)
        metadata_all = x_all[:, -9:].toarray()
        metadata_test = x_test[:, -9:].toarray()

        train_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED,
        )
        clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        metadata_clf = CatBoostClassifier(**CATBOOST_PARAMS)
        started = time.monotonic()
        clf.fit(x_all[train_idx], y[train_idx])
        metadata_clf.fit(metadata_all[train_idx], y[train_idx])
        p_val = (0.5 * clf.predict_proba(x_all[val_idx])[:, 1]
                 + 0.5 * metadata_clf.predict_proba(metadata_all[val_idx])[:, 1])
        # Check the requested holdout blend without computing a forbidden local score.
        assert np.isfinite(p_val).all() and ((p_val >= 0) & (p_val <= 1)).all()
        elapsed = time.monotonic() - started
        tracking.log("log_metric", "training_seconds", elapsed, step=0)
        print(f"Holdout blend ready; training_seconds={elapsed:.3f}", flush=True)

        started = time.monotonic()
        clf.fit(x_all, y)
        metadata_clf.fit(metadata_all, y)
        p = (0.5 * clf.predict_proba(x_test)[:, 1]
             + 0.5 * metadata_clf.predict_proba(metadata_test)[:, 1])
        elapsed = time.monotonic() - started
        tracking.log("log_metric", "training_seconds", elapsed, step=1)
        print(f"Full-data blend ready; training_seconds={elapsed:.3f}", flush=True)
        sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
        sub.to_csv("submission.csv", index=False)
        os.makedirs('submission', exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
        tracking.log("log_metric", "submission_rows", len(sub), step=1)
        print(f"Wrote submission_rows={len(sub)}", flush=True)


if __name__ == "__main__":
    main()
