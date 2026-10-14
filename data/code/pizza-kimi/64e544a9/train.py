"""Metadata-only LightGBM for random-acts-of-pizza (L1-6e9c-1).

Drops the text pipeline (TfidfVectorizer + TruncatedSVD) entirely: the design
matrix is only the nine requester-history numeric columns (clip at 0, log1p,
StandardScaler fit on the training split). Model and protocol are identical to
the parent baseline: LGBMClassifier with boosting rounds selected by 5-fold
stratified CV with early stopping on mean AUC, refit on full train, predict
probabilities on test.

Motivation: with ~5.7k training rows, surface wording is the dominant source of
variance, whereas requester status/history (account age, activity, subreddit
breadth, karma) are the stable behavioral drivers identified by Althoff et al.
(ICWSM 2014). A low-dimensional behavioral model should generalize more
reliably and trains orders of magnitude faster.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
NUM_COLS = ["requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request"]

LGBM_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
                   n_jobs=4, verbose=-1)


# --- MLflow telemetry: best-effort, never allowed to kill training -------------
class _NoRun:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


try:
    import mlflow
    _mlflow_ok = True
except Exception as e:  # pragma: no cover
    print("mlflow unavailable, tracking disabled: %s" % e)
    _mlflow_ok = False


def start_run():
    if not _mlflow_ok:
        return _NoRun()
    try:
        return mlflow.start_run()
    except Exception as e:
        print("mlflow start_run failed, tracking disabled: %s" % e)
        return _NoRun()


def log_params(d):
    if _mlflow_ok:
        try:
            mlflow.log_params(d)
        except Exception as e:
            print("mlflow log_params failed: %s" % e)


def log_metric(name, value):
    if _mlflow_ok:
        try:
            mlflow.log_metric(name, value)
        except Exception as e:
            print("mlflow log_metric failed: %s" % e)
# -------------------------------------------------------------------------------


def load(name):
    with open(os.path.join(PUBLIC, name)) as f:
        return pd.DataFrame(json.load(f))


def featurize(train, test):
    """Return the scaled numeric design matrices for train and test."""
    scaler = StandardScaler()
    x_all = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    x_test = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    return x_all, x_test


def main():
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)
    print("design matrix: %s train, %s test" % (x_all.shape, x_test.shape))

    with start_run():
        log_params({"model": "LGBMClassifier", "features": "metadata_only",
                    "n_numeric_features": len(NUM_COLS), **LGBM_PARAMS})

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cvres = lgb.cv({**LGBM_PARAMS, "metric": "auc"},
                       lgb.Dataset(x_all, label=y),
                       num_boost_round=1500, folds=cv,
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs = cvres["valid auc-mean"]
        best_iteration = int(np.argmax(aucs)) + 1
        best_auc = float(max(aucs))
        print("best_iteration=%d" % best_iteration)
        print("local_cv_auc=%.5f" % best_auc)
        log_metric("best_iteration", best_iteration)
        log_metric("local_cv_auc", best_auc)

        clf = lgb.LGBMClassifier(n_estimators=best_iteration, **LGBM_PARAMS)
        clf.fit(x_all, y)
        p = clf.predict_proba(x_test)[:, 1]

    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
