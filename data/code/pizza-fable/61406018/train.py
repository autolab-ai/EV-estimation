"""Baseline for random-acts-of-pizza.

Word TF-IDF over the request text plus the nine requester-history numbers that the test
split also carries, fed to a plain logistic regression. Targets the historical leaderboard
median (~0.60 ROC AUC); deliberately NOT state of the art: unigrams only, no character
n-grams, no engineered text statistics, no cross-validated blending, no calibration.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 0.3                      # modest, untuned regularisation
MAX_FEATURES = 5_000         # unigrams only (vocabulary for the topic model)
N_TOPICS = 50                # LDA topic proportions replace the sparse TF-IDF block
LDA_ITER = 25
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


def giver_known(df):
    """1 if a giver username is recorded for the row, else 0 (zeros if the column is absent)."""
    if "giver_username_if_known" not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    vals = df["giver_username_if_known"].map(lambda v: str(v).strip())
    return (~vals.isin(["N/A", "", "None"])).to_numpy(dtype=np.float64)


def featurize(train, test):
    vec = CountVectorizer(ngram_range=(1, 1), min_df=3, max_features=MAX_FEATURES, stop_words="english")
    ct = vec.fit_transform(text_of(train))
    cs = vec.transform(text_of(test))
    lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="batch", max_iter=LDA_ITER,
                                    random_state=SEED, n_jobs=-1)
    tt = lda.fit_transform(ct)                      # 50-dim document-topic proportions
    ts = lda.transform(cs)
    tscaler = StandardScaler()
    xt = csr_matrix(tscaler.fit_transform(tt))
    xs = csr_matrix(tscaler.transform(ts))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    gt = csr_matrix(giver_known(train).reshape(-1, 1))   # unscaled 0/1 indicator
    gs = csr_matrix(giver_known(test).reshape(-1, 1))
    return hstack([xt, csr_matrix(nt), gt]).tocsr(), hstack([xs, csr_matrix(ns), gs]).tocsr()


class _Tracker:
    """Best-effort MLflow logging: disables itself after the first failure."""

    def __init__(self):
        self.ok = False
        self.run = None
        try:
            import mlflow
            self.mlflow = mlflow
            self.run = mlflow.start_run()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print("mlflow disabled: %s" % e)

    def params(self, d):
        if self.ok:
            try:
                self.mlflow.log_params(d)
            except Exception as e:  # noqa: BLE001
                print("mlflow disabled: %s" % e); self.ok = False

    def metric(self, k, v, step=0):
        if self.ok:
            try:
                self.mlflow.log_metric(k, float(v), step=step)
            except Exception as e:  # noqa: BLE001
                print("mlflow disabled: %s" % e); self.ok = False

    def end(self):
        if self.ok:
            try:
                self.mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


def main():
    trk = _Tracker()
    trk.params({"C": C, "max_features": MAX_FEATURES, "n_topics": N_TOPICS, "lda_iter": LDA_ITER,
                "text_repr": "lda50", "seed": SEED})
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)
    g = giver_known(train)
    print("giver_known_sum=%d giver_known_pos_rate=%.4f" % (g.sum(), y[g == 1].mean() if g.sum() else float("nan")))

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    val_auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
    print("local_holdout_auc=%.5f" % val_auc)
    trk.metric("local_holdout_auc", val_auc)
    trk.metric("n_features", x_all.shape[1])

    clf.fit(x_all, y)
    p = clf.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    trk.end()


if __name__ == "__main__":
    main()
