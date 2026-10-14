"""Dense-only semantic model for random-acts-of-pizza (L1-d51c-1).

Same as the root baseline but with the 5,000-dimensional word TF-IDF block removed:
the design matrix is just the nine log1p-transformed, standardized requester-history
numeric columns plus the 384-dimensional all-MiniLM-L6-v2 sentence embeddings
(normalized, then standardized), assembled as a plain float64 numpy array (393 dense
features). Motivation: with ~4.5k training documents the sparse unigram block lets
idiosyncratic rare tokens soak up coefficient mass and adds variance a small linear
model cannot average away, while the pretrained encoder has already distilled general
request semantics from far more text than this dataset contains.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 0.3                      # modest, untuned regularisation
NUM_COLS = ["requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request"]

# --- best-effort MLflow tracking: never let telemetry kill training -------------
try:
    import mlflow  # noqa: F401
    _MLFLOW_OK = True
except Exception as _e:  # pragma: no cover
    print("mlflow unavailable, tracking disabled: %s" % _e)
    _MLFLOW_OK = False


def log_params_safe(params):
    if not _MLFLOW_OK:
        return
    try:
        mlflow.log_params(params)
    except Exception as e:
        print("mlflow.log_params failed, continuing untracked: %s" % e)


def log_metric_safe(name, value, step=None):
    if not _MLFLOW_OK:
        return
    try:
        mlflow.log_metric(name, value, step=step)
    except Exception as e:
        print("mlflow.log_metric failed, continuing untracked: %s" % e)


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

    # Nine requester-history numerics, log1p-transformed and standardized.
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))

    # Pretrained sentence embeddings (all-MiniLM-L6-v2, CPU).
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

    # Dense-only design matrix: 9 numerics + 384 embedding dims = 393 features.
    return (np.hstack([nt, emb_tr]).astype(np.float64),
            np.hstack([ns, emb_te]).astype(np.float64))


def main():
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    try:
        run_ctx = mlflow.start_run() if _MLFLOW_OK else _NullCtx()
    except Exception as e:
        print("mlflow.start_run failed, continuing untracked: %s" % e)
        run_ctx = _NullCtx()

    with run_ctx:
        log_params_safe({"model": "logreg_dense_only", "C": C, "max_iter": 2000,
                         "seed": SEED, "features": "9_numeric+384_minilm", "tfidf": "removed"})

        x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
        clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        clf.fit(x_tr, y_tr)
        holdout_auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
        print("local_holdout_auc=%.5f" % holdout_auc)
        log_metric_safe("local_holdout_auc", holdout_auc)

        clf.fit(x_all, y)
        p = clf.predict_proba(x_test)[:, 1]
        sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
