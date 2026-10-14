"""Word unigram/bigram TF-IDF plus nine requester-history features.

Writes submission.csv and submission/submission.csv; only the evaluator scores them.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler

from tracking import tracking_run

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
ALPHA = 1.0
BINARIZE = 0.0
FIT_PRIOR = True
MAX_FEATURES = 5_000         # shared budget for unigrams and bigrams
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
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    return hstack([xt, csr_matrix(nt)]).tocsr(), hstack([xs, csr_matrix(ns)]).tocsr()


def main():
    params = {"ngram_range": "1,2", "min_df": 3, "max_features": MAX_FEATURES,
              "sublinear_tf": True, "stop_words": "english", "numeric_features": len(NUM_COLS),
              "alpha": ALPHA, "binarize": BINARIZE, "fit_prior": FIT_PRIOR}
    with tracking_run(params) as log_metric:
        train, test = load("train.json"), load("test.json")
        y = train["requester_received_pizza"].astype(int).to_numpy()
        started = time.monotonic()
        x_all, x_test = featurize(train, test)
        log_metric("feature_seconds", time.monotonic() - started)
        log_metric("feature_count", x_all.shape[1])

        clf = BernoulliNB(alpha=ALPHA, binarize=BINARIZE, fit_prior=FIT_PRIOR)
        started = time.monotonic()
        clf.fit(x_all, y)
        log_metric("training_seconds", time.monotonic() - started)
        p = clf.predict_proba(x_test)[:, 1]
        sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
        sub.to_csv("submission.csv", index=False)
        os.makedirs('submission', exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
        log_metric("prediction_count", len(sub))
        print("Submission written.", flush=True)


if __name__ == "__main__":
    main()
