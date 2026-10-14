"""Baseline for random-acts-of-pizza.

Word TF-IDF over the request text plus the nine requester-history numbers that the test
split also carries, fed to a plain logistic regression. Targets the historical leaderboard
median (~0.60 ROC AUC); deliberately NOT state of the art: unigrams only, no character
n-grams, no engineered text statistics, no cross-validated blending, no calibration.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class _Tracker:
    """Best-effort MLflow logging; disables itself on the first failure."""

    def __init__(self):
        self.ok = True
        try:
            import mlflow  # noqa
            self.mlflow = mlflow
        except Exception as e:  # pragma: no cover
            print("mlflow unavailable: %s" % e)
            self.ok = False

    def start(self):
        if not self.ok:
            return contextlib.nullcontext()
        try:
            return self.mlflow.start_run()
        except Exception as e:
            print("mlflow start_run failed, continuing untracked: %s" % e)
            self.ok = False
            return contextlib.nullcontext()

    def params(self, d):
        if self.ok:
            try:
                self.mlflow.log_params(d)
            except Exception as e:
                print("mlflow log_params failed: %s" % e)
                self.ok = False

    def metric(self, k, v, step=0):
        if self.ok:
            try:
                self.mlflow.log_metric(k, float(v), step=step)
            except Exception as e:
                print("mlflow log_metric failed: %s" % e)
                self.ok = False


COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
N_ESTIMATORS = 500           # random-forest size
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


def dense_block(df):
    """Extra dense features: request time, text statistics, requester-history indicators.

    Returns (array, continuous_mask) where continuous_mask marks columns to be standardized.
    """
    n = len(df)
    cols, cont = [], []

    def add(v, is_cont):
        cols.append(np.asarray(v, dtype=np.float64).reshape(n))
        cont.append(is_cont)

    # (a) time of request
    ts = pd.to_numeric(df["unix_timestamp_of_request_utc"], errors="coerce").fillna(0).astype(float)
    t_years = ts / (365.25 * 86400.0)
    add(t_years, True)
    add(t_years ** 2, True)
    dt = pd.to_datetime(ts, unit="s")
    h = dt.dt.hour.to_numpy()
    add(np.sin(2 * np.pi * h / 24.0), True)
    add(np.cos(2 * np.pi * h / 24.0), True)
    dow = dt.dt.dayofweek.to_numpy()
    for d in range(7):
        add((dow == d).astype(float), False)

    # (b) text statistics
    title = df.get("request_title", pd.Series([""] * n)).fillna("").astype(str)
    body = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * n))).fillna("").astype(str)
    body_l, title_l = body.str.lower(), title.str.lower()
    add(np.log1p(body.str.split().str.len().fillna(0)), True)
    add(np.log1p(body.str.len()), True)
    add(np.log1p(title.str.split().str.len().fillna(0)), True)
    add(np.log1p(body.str.count("http")), True)
    add(body_l.str.contains("imgur", regex=False).astype(float), False)
    add(title_l.str.contains("[request]", regex=False).astype(float), False)
    add(np.log1p(body.str.count(r"!")), True)
    add(np.log1p(body.str.count(r"\?")), True)
    add(np.log1p(body.str.count("\n")), True)
    upper = body.str.count(r"[A-Z]").to_numpy(dtype=float)
    letters = body.str.count(r"[A-Za-z]").to_numpy(dtype=float)
    add(np.where(letters > 0, upper / np.maximum(letters, 1), 0.0), True)

    # (c) requester-history indicators
    posts_raop = pd.to_numeric(df["requester_number_of_posts_on_raop_at_request"], errors="coerce").fillna(0)
    comm_raop = pd.to_numeric(df["requester_number_of_comments_in_raop_at_request"], errors="coerce").fillna(0)
    add((posts_raop > 0).astype(float), False)
    add((comm_raop > 0).astype(float), False)
    minus = pd.to_numeric(df["requester_upvotes_minus_downvotes_at_request"], errors="coerce").fillna(0).to_numpy(dtype=float)
    plus = pd.to_numeric(df["requester_upvotes_plus_downvotes_at_request"], errors="coerce").fillna(0).to_numpy(dtype=float)
    add(minus / np.maximum(1.0, plus), True)

    return np.column_stack(cols), np.array(cont, dtype=bool)


def featurize(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    dt_, cont = dense_block(train)
    ds_, _ = dense_block(test)
    dscaler = StandardScaler()
    dt_[:, cont] = dscaler.fit_transform(dt_[:, cont])
    ds_[:, cont] = dscaler.transform(ds_[:, cont])
    print("dense_block_cols=%d" % dt_.shape[1])
    return (hstack([xt, csr_matrix(nt), csr_matrix(dt_)]).tocsr(),
            hstack([xs, csr_matrix(ns), csr_matrix(ds_)]).tocsr())


def main():
    tracker = _Tracker()
    with tracker.start():
        tracker.params({"model": "random_forest", "n_estimators": N_ESTIMATORS, "max_features": MAX_FEATURES, "seed": SEED, "dense_block": "time+textstats+history"})
        run(tracker)


def run(tracker):
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, criterion="gini", max_features="sqrt", max_depth=None,
                                 min_samples_leaf=1, bootstrap=True, n_jobs=-1, random_state=SEED)
    clf.fit(x_tr, y_tr)
    val_auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
    print("local_holdout_auc=%.5f" % val_auc)
    tracker.metric("local_holdout_auc", val_auc)

    clf.fit(x_all, y)
    p = clf.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
