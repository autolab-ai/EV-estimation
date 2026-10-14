"""Baseline for random-acts-of-pizza.

Word TF-IDF over the request text plus the nine requester-history numbers that the test
split also carries, fed to a plain logistic regression. Targets the historical leaderboard
median (~0.60 ROC AUC); deliberately NOT state of the art: unigrams only, no character
n-grams, no engineered text statistics, no cross-validated blending, no calibration.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

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


def subreddit_lists(df):
    values = df.get("requester_subreddits_at_request", pd.Series([None] * len(df), index=df.index))
    return [
        sorted({name.lower() for name in names if isinstance(name, str)})
        if isinstance(names, list) else []
        for names in values
    ]


def featurize(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    subreddit_vec = CountVectorizer(analyzer=lambda names: names, binary=True, min_df=5)
    rt = subreddit_vec.fit_transform(subreddit_lists(train))
    rs = subreddit_vec.transform(subreddit_lists(test))
    timestamps_train = train["unix_timestamp_of_request_utc"].to_numpy(dtype=np.float64)
    timestamps_test = test["unix_timestamp_of_request_utc"].to_numpy(dtype=np.float64)
    timestamp_median = np.median(timestamps_train[np.isfinite(timestamps_train)])
    days_train = np.where(np.isfinite(timestamps_train), timestamps_train, timestamp_median) / 86400
    days_test = np.where(np.isfinite(timestamps_test), timestamps_test, timestamp_median) / 86400
    date_spline = SplineTransformer(
        n_knots=6, degree=3, knots="quantile", extrapolation="constant", include_bias=False
    )
    date_scaler = StandardScaler()
    dt = date_scaler.fit_transform(date_spline.fit_transform(days_train.reshape(-1, 1)))
    ds = date_scaler.transform(date_spline.transform(days_test.reshape(-1, 1)))
    return (hstack([xt, csr_matrix(nt), rt, csr_matrix(dt)]).tocsr(),
            hstack([xs, csr_matrix(ns), rs, csr_matrix(ds)]).tocsr())


class BestEffortTracking:
    def __init__(self):
        self.client = None
        self.warned = False

    def disable(self, exc):
        self.client = None
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable ({exc}); continuing.", flush=True)
            self.warned = True

    def call(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as exc:
                self.disable(exc)


@contextmanager
def tracking_run():
    tracker = BestEffortTracking()
    run = None
    try:
        import mlflow
        run = mlflow.start_run()
        tracker.client = mlflow
    except Exception as exc:
        tracker.disable(exc)
    try:
        yield tracker
    finally:
        if run is not None:
            try:
                mlflow.end_run()
            except Exception as exc:
                tracker.disable(exc)


def main(tracker):
    tracker.call("log_params", {
        "C": C, "max_iter": 2000, "random_state": SEED,
        "max_features": MAX_FEATURES, "ngram_range": "1,1", "text_min_df": 3,
        "sublinear_tf": True, "stop_words": "english", "numeric_columns": len(NUM_COLS),
        "subreddit_binary": True, "subreddit_min_df": 5, "subreddit_scaled": False,
        "date_column": "unix_timestamp_of_request_utc", "date_units": "days",
        "date_n_knots": 6, "date_degree": 3, "date_knots": "quantile",
        "date_extrapolation": "constant", "date_include_bias": False, "date_scaled": True,
    })
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    start = time.perf_counter()
    x_all, x_test = featurize(train, test)
    tracker.call("log_metric", "feature_seconds", time.perf_counter() - start, step=0)
    tracker.call("log_metric", "n_features", x_all.shape[1], step=0)

    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    start = time.perf_counter()
    clf.fit(x_all, y)
    tracker.call("log_metric", "train_seconds", time.perf_counter() - start, step=0)
    p = clf.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    with tracking_run() as tracker:
        main(tracker)
