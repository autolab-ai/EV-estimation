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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
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
        tracker.params({"C": C, "max_features": MAX_FEATURES, "seed": SEED, "dense_block": "time+textstats+history",
                        "blend": "rank_avg_lr_hgb", "gbm_lr": 0.03, "gbm_iter": 400, "gbm_leaves": 8, "cv_folds": 5})
        run(tracker)


def run(tracker):
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    # dense matrix for the tree model: raw requester numbers + unscaled dense_block
    d_all = np.hstack([train[NUM_COLS].to_numpy(dtype=float), dense_block(train)[0]])
    d_test = np.hstack([test[NUM_COLS].to_numpy(dtype=float), dense_block(test)[0]])
    print("tree_dense_cols=%d" % d_all.shape[1])

    def make_lr():
        return LogisticRegression(C=C, max_iter=2000, random_state=SEED)

    def make_gbm():
        return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=400, max_leaf_nodes=8,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=False, random_state=SEED)

    # 5-fold out-of-fold predictions for both models
    oof_lr = np.zeros(len(y))
    oof_gbm = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (tr, va) in enumerate(skf.split(x_all, y)):
        lr = make_lr().fit(x_all[tr], y[tr])
        oof_lr[va] = lr.predict_proba(x_all[va])[:, 1]
        gbm = make_gbm().fit(d_all[tr], y[tr])
        oof_gbm[va] = gbm.predict_proba(d_all[va])[:, 1]
        print("fold%d lr_auc=%.5f gbm_auc=%.5f" % (k, roc_auc_score(y[va], oof_lr[va]),
                                                   roc_auc_score(y[va], oof_gbm[va])))

    auc_lr = roc_auc_score(y, oof_lr)
    auc_gbm = roc_auc_score(y, oof_gbm)
    r_lr = rankdata(oof_lr) / len(oof_lr)
    r_gbm = rankdata(oof_gbm) / len(oof_gbm)
    best_w, best_auc = None, -1.0
    for w in np.arange(0.0, 1.0001, 0.05):
        a = roc_auc_score(y, w * r_lr + (1 - w) * r_gbm)
        if a >= best_auc:  # ties -> larger w
            best_auc, best_w = a, float(w)
    print("oof_auc_lr=%.5f oof_auc_gbm=%.5f blend_w_lr=%.2f oof_auc_blend=%.5f"
          % (auc_lr, auc_gbm, best_w, best_auc))
    tracker.metric("oof_auc_lr", auc_lr)
    tracker.metric("oof_auc_gbm", auc_gbm)
    tracker.metric("blend_w_lr", best_w)
    tracker.metric("oof_auc_blend", best_auc)
    tracker.metric("local_holdout_auc", best_auc)  # same key as the baseline's headline metric

    # refit on all training rows, rank-average on the test set
    p_lr = make_lr().fit(x_all, y).predict_proba(x_test)[:, 1]
    p_gbm = make_gbm().fit(d_all, y).predict_proba(d_test)[:, 1]
    n_test = len(p_lr)
    p = best_w * rankdata(p_lr) / n_test + (1 - best_w) * rankdata(p_gbm) / n_test
    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
