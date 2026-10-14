"""Baseline for random-acts-of-pizza.

Word TF-IDF over the request text plus the nine requester-history numbers that the test
split also carries, fed to a plain logistic regression. Targets the historical leaderboard
median (~0.60 ROC AUC); deliberately NOT state of the art: unigrams only, no character
n-grams, no engineered text statistics, no cross-validated blending, no calibration.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def subreddit_docs(df):
    """One document per row: the requester's prior subreddits, lowercased, space-joined."""
    col = df.get("requester_subreddits_at_request", pd.Series([None] * len(df)))

    def to_doc(v):
        if v is None or not isinstance(v, (list, tuple)) or len(v) == 0:
            return ""
        return " ".join(str(s).lower() for s in v if s is not None)

    return col.apply(to_doc).astype(str)

FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|me|my|mine|myself)\b")
RECIPROCITY_RE = re.compile(r"pay (it|this|you|the favou?r) (forward|back)|return the favou?r|repay|pay ?back|when i get paid|payday")
GRATITUDE_RE = re.compile(r"\bthank|appreciat|grateful|bless")
URL_RE = re.compile(r"https?://|www\.", re.I)
IMAGE_RE = re.compile(r"imgur|\.jpe?g\b|\.png\b|\.gif\b", re.I)
MONEY_RE = re.compile(r"\$\s?\d|\d+\s?(dollars|bucks)", re.I)


def dense_feats(df, ts_fill=None):
    """31 dense columns: temporal, request-form and requester-history flags.

    ts_fill: value used for missing timestamps (train median); returned for reuse on test.
    """
    n = len(df)
    if "unix_timestamp_of_request_utc" in df.columns:
        ts = pd.to_numeric(df["unix_timestamp_of_request_utc"], errors="coerce").astype(float)
    else:
        ts = pd.to_numeric(df["unix_timestamp_of_request"], errors="coerce").astype(float)
    if ts_fill is None:
        ts_fill = float(ts.median())
    ts = ts.fillna(ts_fill)
    dt = pd.to_datetime(ts, unit="s", utc=True)
    cols = []
    cols.append(ts.to_numpy() / 86400.0)                                  # 1 t_days
    hour_bin = (dt.dt.hour // 4).to_numpy()
    for b in range(6):                                                    # 2-7 hour bins
        cols.append((hour_bin == b).astype(float))
    wd = dt.dt.dayofweek.to_numpy()
    for d in range(7):                                                    # 8-14 weekday
        cols.append((wd == d).astype(float))
    cols.append(dt.dt.day.to_numpy().astype(float))                       # 15 day_of_month

    body = df.get("request_text_edit_aware", pd.Series([""] * n)).fillna("").astype(str)
    title = df.get("request_title", pd.Series([""] * n)).fillna("").astype(str)
    body_lower = body.str.lower()
    n_words_body = body.apply(lambda s: len(s.split())).to_numpy().astype(float)
    cols.append(np.log1p(n_words_body))                                   # 16
    cols.append(np.log1p(title.apply(lambda s: len(s.split())).to_numpy().astype(float)))  # 17
    cols.append(np.log1p(body.str.len().to_numpy().astype(float)))        # 18
    cols.append(body.apply(lambda s: 1.0 if URL_RE.search(s) else 0.0).to_numpy())    # 19
    cols.append(body.apply(lambda s: 1.0 if IMAGE_RE.search(s) else 0.0).to_numpy())  # 20
    cols.append(body.apply(lambda s: 1.0 if MONEY_RE.search(s) else 0.0).to_numpy())  # 21
    cols.append(np.log1p(body.str.count("!").to_numpy().astype(float)))   # 22
    cols.append(np.log1p(body.str.count(r"\?").to_numpy().astype(float)))  # 23
    n_upper = body.apply(lambda s: sum(c.isupper() for c in s)).to_numpy().astype(float)
    n_alpha = body.apply(lambda s: sum(c.isalpha() for c in s)).to_numpy().astype(float)
    cols.append(n_upper / np.maximum(1.0, n_alpha))                       # 24 caps_ratio
    fp = body_lower.apply(lambda s: len(FIRST_PERSON_RE.findall(s))).to_numpy().astype(float)
    cols.append(fp / np.maximum(1.0, n_words_body))                       # 25 first_person_rate
    cols.append(body_lower.apply(lambda s: 1.0 if RECIPROCITY_RE.search(s) else 0.0).to_numpy())  # 26
    cols.append(body_lower.apply(lambda s: 1.0 if GRATITUDE_RE.search(s) else 0.0).to_numpy())    # 27

    def num(col):
        return pd.to_numeric(df.get(col, pd.Series([0] * n)), errors="coerce").fillna(0).to_numpy().astype(float)

    cols.append((num("requester_number_of_posts_on_raop_at_request") > 0).astype(float))     # 28
    cols.append((num("requester_number_of_comments_in_raop_at_request") > 0).astype(float))  # 29
    cols.append((num("requester_account_age_in_days_at_request") < 1).astype(float))         # 30
    plus = num("requester_upvotes_plus_downvotes_at_request")
    minus = num("requester_upvotes_minus_downvotes_at_request")
    ratio = np.where(plus > 0, minus / np.where(plus > 0, plus, 1.0), 0.0)
    cols.append(np.clip(ratio, -1.0, 1.0))                                # 31 upvote_ratio
    arr = np.column_stack(cols).astype(np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr, ts_fill


def featurize(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    # bag-of-subreddits block: binary presence, idf-weighted, l2-normalised per row
    svec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=True, binary=True,
                           use_idf=True, norm="l2", min_df=5, max_features=3000)
    st = svec.fit_transform(subreddit_docs(train))
    ss = svec.transform(subreddit_docs(test))
    print("subreddit_vocab=%d" % len(svec.vocabulary_))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    dense_train, ts_fill = dense_feats(train)
    dense_test, _ = dense_feats(test, ts_fill=ts_fill)
    dscaler = StandardScaler()
    dense_train = dscaler.fit_transform(dense_train)
    dense_test = dscaler.transform(dense_test)
    print("dense_block_cols=%d" % dense_train.shape[1])
    return (hstack([xt, st, csr_matrix(nt), csr_matrix(dense_train)]).tocsr(),
            hstack([xs, ss, csr_matrix(ns), csr_matrix(dense_test)]).tocsr())


def main():
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    print("local_holdout_auc=%.5f" % roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1]))

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
