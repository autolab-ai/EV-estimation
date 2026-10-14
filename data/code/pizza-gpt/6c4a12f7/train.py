"""Baseline for random-acts-of-pizza.

Word TF-IDF over the request text plus the nine requester-history numbers that the test
split also carries, fed to a plain logistic regression. Targets the historical leaderboard
median (~0.60 ROC AUC); deliberately NOT state of the art: unigrams only, no character
n-grams, no engineered text statistics, no cross-validated blending, no calibration.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer, StandardScaler

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 30.0                     # weaker shared L2 regularisation
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


class BestEffortTracking:
    """Attach to the evaluator's configured run without risking training."""

    def __init__(self):
        self.enabled = False
        self.warned = False
        self.context = None

    def warn(self, error):
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {error}", file=sys.stderr, flush=True)
            self.warned = True
        self.enabled = False

    def __enter__(self):
        try:
            import mlflow
            self.mlflow = mlflow
            context = mlflow.start_run()
            context.__enter__()
            self.context = context
            self.enabled = True
        except Exception as error:
            self.warn(error)
        return self

    def call(self, method, *args, **kwargs):
        if self.enabled:
            try:
                getattr(self.mlflow, method)(*args, **kwargs)
            except Exception as error:
                self.warn(error)

    def log_params(self, params):
        self.call("log_params", params)

    def log_metric(self, name, value, step=0):
        self.call("log_metric", name, value, step=step)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc_value, traceback)
            except Exception as error:
                self.warn(error)
        return False


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


def date_features(train, test):
    """Fit date imputation and spline knots on train, then transform both sets."""
    column = "unix_timestamp_of_request_utc"
    train_dates = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=np.float64)
    test_dates = pd.to_numeric(test[column], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(train_dates)
    if not finite.any():
        raise ValueError("Training partition has no finite UTC request timestamps")
    median = np.median(train_dates[finite])
    train_days = (np.where(finite, train_dates, median) / 86400.0).reshape(-1, 1)
    test_days = (np.where(np.isfinite(test_dates), test_dates, median) / 86400.0).reshape(-1, 1)
    spline = SplineTransformer(n_knots=6, degree=3, knots="quantile",
                               include_bias=False, extrapolation="constant")
    return csr_matrix(spline.fit_transform(train_days)), csr_matrix(spline.transform(test_days))


def main(tracking):
    started = time.perf_counter()
    tracking.log_params({"C": C, "max_iter": 2000, "random_state": SEED,
                         "max_features": MAX_FEATURES, "min_df": 3,
                         "ngram_range": "1,1", "sublinear_tf": True,
                         "stop_words": "english", "numeric_features": len(NUM_COLS),
                         "date_column": "unix_timestamp_of_request_utc",
                         "date_imputation": "training_median", "date_units": "days",
                         "spline_n_knots": 6, "spline_degree": 3,
                         "spline_knots": "quantile", "spline_include_bias": False,
                         "spline_extrapolation": "constant"})
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    # Preserve the baseline's word and numeric feature construction.
    x_all, x_test = featurize(train, test)

    tr_idx, val_idx = train_test_split(np.arange(len(train)), test_size=0.2,
                                     stratify=y, random_state=SEED)
    dt, dv = date_features(train.iloc[tr_idx], train.iloc[val_idx])
    x_tr = hstack([x_all[tr_idx], dt], format="csr")
    x_val = hstack([x_all[val_idx], dv], format="csr")
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    fit_started = time.perf_counter()
    clf.fit(x_tr, y[tr_idx])
    # Exercise holdout inference without computing or printing a score.
    assert np.isfinite(clf.predict_proba(x_val)).all()
    tracking.log_metric("fit_seconds", time.perf_counter() - fit_started, step=0)
    tracking.log_metric("solver_iterations", int(clf.n_iter_.max()), step=0)
    print("Holdout fit and inference complete (no score computed).", flush=True)

    # Refit date preprocessing on every labeled row for the final model.
    da, ds = date_features(train, test)
    x_all = hstack([x_all, da], format="csr")
    x_test = hstack([x_test, ds], format="csr")
    fit_started = time.perf_counter()
    clf.fit(x_all, y)
    tracking.log_metric("fit_seconds", time.perf_counter() - fit_started, step=1)
    tracking.log_metric("solver_iterations", int(clf.n_iter_.max()), step=1)
    p = clf.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    elapsed = time.perf_counter() - started
    tracking.log_metric("training_seconds", elapsed, step=1)
    tracking.log_metric("feature_count", x_all.shape[1], step=1)
    tracking.log_metric("submission_rows", len(sub), step=1)
    print(f"Wrote {len(sub)} predictions with {x_all.shape[1]} features in {elapsed:.2f}s.", flush=True)


if __name__ == "__main__":
    with BestEffortTracking() as tracking:
        main(tracking)
