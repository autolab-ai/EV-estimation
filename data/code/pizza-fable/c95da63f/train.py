"""Two-level stack for random-acts-of-pizza (child of the baseline).

Level 1: word TF-IDF (same configuration as the baseline) -> logistic regression, fit
out-of-fold so every train row gets a text logit it was not trained on; test logits are the
average over the five fold models.
Level 2: LightGBM on a dense 11-column table: the text logit, the nine raw requester-history
numbers and the raw UTC request timestamp. Early-stopped 5-fold CV picks the tree count;
the final model is refit on all rows with three seeds and averaged.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 0.3                      # level-1 regularisation (unchanged from baseline)
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
TS_COL = "unix_timestamp_of_request_utc"
LGB_PARAMS = dict(objective="binary", learning_rate=0.01, n_estimators=3000, num_leaves=8,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=10.0, n_jobs=int(os.environ.get("LGB_THREADS", "4")), verbose=-1)
FINAL_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- best-effort MLflow tracking
class _Tracker:
    def __init__(self):
        self.ok = False
        self.mlflow = None
        try:
            import mlflow  # noqa: WPS433
            self.mlflow = mlflow
            self.ok = True
        except Exception as e:  # pragma: no cover
            print("mlflow unavailable: %s" % e)

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
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:
            print("mlflow log_params failed, disabling tracking: %s" % e)
            self.ok = False

    def metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print("mlflow log_metric failed, disabling tracking: %s" % e)
            self.ok = False


def load(name):
    with open(os.path.join(PUBLIC, name)) as f:
        return pd.DataFrame(json.load(f))


def text_of(df):
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("")
    body = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * len(df)))).fillna("")
    return (title + " " + body).astype(str)


def text_level(train, test, y, trk):
    """Out-of-fold text logits for train, fold-averaged logits for test."""
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True,
                          max_features=MAX_FEATURES, stop_words="english")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    text_oof = np.zeros(len(train), dtype=np.float64)
    text_test = np.zeros(xs.shape[0], dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for i, (tr, va) in enumerate(skf.split(xt, y)):
        clf = LogisticRegression(C=C, max_iter=2000, random_state=0)
        clf.fit(xt[tr], y[tr])
        text_oof[va] = clf.decision_function(xt[va])
        text_test += clf.decision_function(xs) / skf.n_splits
        trk.metric("text_fold_auc", roc_auc_score(y[va], text_oof[va]), step=i)
    text_auc = roc_auc_score(y, text_oof)
    print("text_oof_auc=%.5f" % text_auc)
    trk.metric("text_oof_auc", text_auc)
    return text_oof, text_test


def meta_table(df, text_score):
    cols = [text_score.reshape(-1, 1),
            df[NUM_COLS].to_numpy(dtype=np.float64),
            df[TS_COL].to_numpy(dtype=np.float64).reshape(-1, 1)]
    return np.hstack(cols)


def main():
    trk = _Tracker()
    with trk.start():
        trk.params(dict(model="stack_tfidf_lr_to_lgbm", C=C, max_features=MAX_FEATURES,
                        **{"lgb_" + k: v for k, v in LGB_PARAMS.items() if k not in ("verbose",)},
                        final_seeds=len(FINAL_SEEDS)))
        train, test = load("train.json"), load("test.json")
        y = train["requester_received_pizza"].astype(int).to_numpy()

        text_oof, text_test = text_level(train, test, y, trk)
        m_tr = meta_table(train, text_oof)
        m_te = meta_table(test, text_test)

        oof = np.zeros(len(train), dtype=np.float64)
        best_iters = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for i, (tr, va) in enumerate(skf.split(m_tr, y)):
            model = lgb.LGBMClassifier(**LGB_PARAMS)
            model.fit(m_tr[tr], y[tr], eval_set=[(m_tr[va], y[va])], eval_metric="auc",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
            best_iters.append(int(model.best_iteration_))
            oof[va] = model.predict_proba(m_tr[va])[:, 1]
            fold_auc = roc_auc_score(y[va], oof[va])
            trk.metric("fold_auc", fold_auc, step=i)
            trk.metric("best_iteration", best_iters[-1], step=i)
            print("fold%d auc=%.5f best_iter=%d" % (i, fold_auc, best_iters[-1]))
        oof_auc = roc_auc_score(y, oof)
        print("local_oof_auc=%.5f" % oof_auc)
        trk.metric("local_holdout_auc", oof_auc)

        n_final = int(1.1 * float(np.mean(best_iters)))
        print("n_final=%d" % n_final)
        trk.metric("n_final", n_final)
        p = np.zeros(m_te.shape[0], dtype=np.float64)
        for s in FINAL_SEEDS:
            params = dict(LGB_PARAMS, n_estimators=n_final, random_state=s)
            model = lgb.LGBMClassifier(**params)
            model.fit(m_tr, y)
            p += model.predict_proba(m_te)[:, 1] / len(FINAL_SEEDS)

        sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
