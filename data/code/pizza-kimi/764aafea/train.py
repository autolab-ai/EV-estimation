"""TruncatedSVD + RandomForest for random-acts-of-pizza (L1-da1e-1).

Same featurization as the baseline (unigram TF-IDF over title + edit-aware text,
TruncatedSVD(300), nine log1p + StandardScaler requester-history numbers -> 309-dim
dense matrix), but the LightGBM classifier is swapped for a bagged random forest
(sklearn RandomForestClassifier, 400 trees, max_features='sqrt'). A forest has no
boosting rounds to tune, so the lgb.cv early-stopping step is replaced by a 5-fold
stratified cross_val_predict AUC on the training data, computed purely for reporting;
the final forest is fit once on the full training matrix.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
MAX_FEATURES = 5_000         # unigrams only
N_SVD = 300
NUM_COLS = ["requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request"]

RF_PARAMS = dict(n_estimators=400, max_features="sqrt", n_jobs=4,
                 random_state=SEED)


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


def text_of(df):
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("")
    body = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * len(df)))).fillna("")
    return (title + " " + body).astype(str)


def featurize(train, test):
    """Return the TF-IDF matrices and the scaled numeric matrices separately."""
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    ns = scaler.transform(np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)))
    return xt, xs, nt, ns


def main():
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    xt, xs, nt, ns = featurize(train, test)

    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
    x_all = np.hstack([svd.fit_transform(xt), nt])
    x_test = np.hstack([svd.transform(xs), ns])
    print("design matrix: %s train, %s test" % (x_all.shape, x_test.shape))

    with start_run():
        log_params({"model": "RandomForestClassifier", "n_svd": N_SVD,
                    "max_features": MAX_FEATURES, **RF_PARAMS})

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = cross_val_predict(RandomForestClassifier(**RF_PARAMS), x_all, y,
                                cv=cv, method="predict_proba")[:, 1]
        best_auc = float(roc_auc_score(y, oof))
        print("local_cv_auc=%.5f" % best_auc)
        log_metric("local_cv_auc", best_auc)

        clf = RandomForestClassifier(**RF_PARAMS)
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
