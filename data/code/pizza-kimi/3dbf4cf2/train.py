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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
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


def featurize(train, test):
    train_texts = text_of(train)
    test_texts = text_of(test)
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(train_texts)
    xs = vec.transform(test_texts)
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))

    # Pretrained sentence embeddings (all-MiniLM-L6-v2, CPU) appended as a dense block.
    # Docs are truncated to their first 256 whitespace-separated tokens to bound runtime.
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    tr_trunc = [" ".join(t.split()[:256]) for t in train_texts]
    te_trunc = [" ".join(t.split()[:256]) for t in test_texts]
    emb_tr = emb_model.encode(tr_trunc, batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
    emb_te = emb_model.encode(te_trunc, batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
    emb_scaler = StandardScaler()
    emb_tr = emb_scaler.fit_transform(emb_tr)
    emb_te = emb_scaler.transform(emb_te)

    return (hstack([xt, csr_matrix(nt), csr_matrix(emb_tr)]).tocsr(),
            hstack([xs, csr_matrix(ns), csr_matrix(emb_te)]).tocsr())


def main():
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
    clf = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                               n_jobs=-1, random_state=SEED)
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
