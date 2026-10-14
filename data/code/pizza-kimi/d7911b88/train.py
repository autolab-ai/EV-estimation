"""Metadata-only logistic regression for random-acts-of-pizza.

Drops the 100k-dim TF-IDF text block entirely: the model sees only the nine
requester-history numbers that the test split also carries (log1p, standardized),
fed to a plain logistic regression. Motivation: text learned on 2010-2013 wording
drifts and dominates the variance against only ~4.5k training rows, while the
requester-history signals highlighted by Althoff et al. are stable and timestamped
at request time.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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


def load(name):
    with open(os.path.join(PUBLIC, name)) as f:
        return pd.DataFrame(json.load(f))


def featurize(train, test):
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    return csr_matrix(nt), csr_matrix(ns)


def main():
    # best-effort telemetry: never let tracking failures kill the run
    try:
        import mlflow
        run_ctx = mlflow.start_run()
        log_metric = mlflow.log_metric
        mlflow.log_params({"model": "logreg_meta_only", "C": C, "seed": SEED,
                           "features": len(NUM_COLS), "text": "none"})
    except Exception as e:
        print("mlflow disabled: %s" % e)
        from contextlib import nullcontext
        run_ctx = nullcontext()

        def log_metric(*a, **k):
            pass

    with run_ctx:
        train, test = load("train.json"), load("test.json")
        y = train["requester_received_pizza"].astype(int).to_numpy()
        x_all, x_test = featurize(train, test)

        x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
        clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        clf.fit(x_tr, y_tr)
        auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
        print("local_holdout_auc=%.5f" % auc)
        try:
            log_metric("local_holdout_auc", auc)
        except Exception:
            pass

        clf.fit(x_all, y)
        p = clf.predict_proba(x_test)[:, 1]
        sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
