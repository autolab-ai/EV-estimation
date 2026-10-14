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
MAX_FEATURES = 60_000        # unigrams + bigrams
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


def text_stats(df):
    """15 dense per-request statistics computed from the title and body strings."""
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("").astype(str)
    body = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * len(df)))).fillna("").astype(str)
    full = title + " " + body

    title_wc = title.str.split().str.len().to_numpy(dtype=np.float64)                 # 1
    body_cc = body.str.len().to_numpy(dtype=np.float64)                               # 2
    body_wc = body.str.split().str.len().to_numpy(dtype=np.float64)                   # 3
    sent_ct = body.str.count(r"[.!?]").to_numpy(dtype=np.float64)                     # 4
    nonspace = body.str.count(r"\S").to_numpy(dtype=np.float64)
    mean_wlen = nonspace / np.maximum(body_wc, 1.0)                                   # 5
    excl = body.str.count("!").to_numpy(dtype=np.float64)                             # 6
    quest = body.str.count(r"\?").to_numpy(dtype=np.float64)                          # 7
    letters = body.str.count(r"[A-Za-z]").to_numpy(dtype=np.float64)
    upper = body.str.count(r"[A-Z]").to_numpy(dtype=np.float64)
    upper_frac = np.where(letters > 0, upper / np.maximum(letters, 1.0), 0.0)         # 8
    digits = body.str.count(r"\d").to_numpy(dtype=np.float64)                         # 9
    pronouns = full.str.count(r"\b(i|me|my|mine|we|us|our)\b",
                              flags=re.I).to_numpy(dtype=np.float64)                  # 10
    gratitude = full.str.count(r"\b(thank|thanks|grateful|appreciate|appreciated|please)\b",
                               flags=re.I).to_numpy(dtype=np.float64)                 # 11
    recip_phrases = ["pay it forward", "return the favor", "give back", "repay"]
    reciprocity = full.str.count("|".join(re.escape(p) for p in recip_phrases),
                                 flags=re.I).to_numpy(dtype=np.float64)               # 12
    hardship = full.str.count(r"\b(rent|broke|hungry|starving|paycheck|bills|unemployed|evicted)\b",
                              flags=re.I).to_numpy(dtype=np.float64)                  # 13
    money = full.str.count(r"\$\s?\d+").to_numpy(dtype=np.float64)                    # 14
    sentences = np.maximum(sent_ct, 1.0)
    words_safe = np.maximum(body_wc, 1.0)
    syllables = body.str.count(r"[aeiouy]+", flags=re.I).to_numpy(dtype=np.float64)
    flesch = 206.835 - 1.015 * (body_wc / sentences) - 84.6 * (syllables / words_safe)  # 15

    feats = np.column_stack([title_wc, body_cc, body_wc, sent_ct, mean_wlen, excl, quest,
                             upper_frac, digits, pronouns, gratitude, reciprocity,
                             hardship, money, flesch])
    log_idx = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13]   # count features get log1p
    feats[:, log_idx] = np.log1p(feats[:, log_idx])
    return feats


def featurize(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    stat_scaler = StandardScaler()
    st_train = stat_scaler.fit_transform(text_stats(train))
    st_test = stat_scaler.transform(text_stats(test))
    return (hstack([xt, csr_matrix(nt), csr_matrix(st_train)]).tocsr(),
            hstack([xs, csr_matrix(ns), csr_matrix(st_test)]).tocsr())


_MLFLOW_OK = {"run": None, "failed": False}


def _mlflow_start():
    """Attach to the preconfigured run if mlflow is available; never raise."""
    try:
        import mlflow
        _MLFLOW_OK["run"] = mlflow.start_run()
    except Exception as e:  # noqa: BLE001 - telemetry must never kill the run
        print("mlflow disabled: %s" % e)
        _MLFLOW_OK["failed"] = True


def _mlflow_log_params(params):
    if _MLFLOW_OK["run"] is None or _MLFLOW_OK["failed"]:
        return
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception as e:  # noqa: BLE001
        print("mlflow log_params failed: %s" % e)
        _MLFLOW_OK["failed"] = True


def _mlflow_log_metric(name, value, step=None):
    if _MLFLOW_OK["run"] is None or _MLFLOW_OK["failed"]:
        return
    try:
        import mlflow
        mlflow.log_metric(name, value, step=step)
    except Exception as e:  # noqa: BLE001
        print("mlflow log_metric failed: %s" % e)
        _MLFLOW_OK["failed"] = True


def main():
    _mlflow_start()
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
    print("local_holdout_auc=%.5f" % auc)
    _mlflow_log_params({"C": C, "max_features": MAX_FEATURES, "ngram_range": "1,2",
                        "min_df": 2, "sublinear_tf": True, "seed": SEED})
    _mlflow_log_metric("local_holdout_auc", auc)

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
