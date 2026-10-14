"""LSA baseline for jigsaw-toxic-comment-classification-challenge.

A single dense latent-semantic pipeline, replacing the previous two-branch
sparse n-gram rank-average ensemble:
  word 1-2-gram TF-IDF (100k) -> TruncatedSVD(300) -> L2 Normalizer
  -> one LogisticRegression(C=1.0, lbfgs) per label on the dense 300-d features

The vectorizer and the SVD are fit on train+test text, as the previous
vectorizers were.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer

COMP = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 0
N_COMPONENTS = int(os.environ.get("LSA_N_COMPONENTS", "300"))

# ---------------------------------------------------------------- MLflow (best effort)
_TRACK = {"on": True}


def _start_run():
    """Attach to the preconfigured MLflow run; never fail the training run."""
    try:
        import mlflow  # noqa: F401

        return mlflow.start_run()
    except Exception as exc:  # pragma: no cover - telemetry only
        print(f"[mlflow] disabled: {exc}", flush=True)
        _TRACK["on"] = False
        return contextlib.nullcontext()


def log_params(params):
    if not _TRACK["on"]:
        return
    try:
        import mlflow

        mlflow.log_params(params)
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] log_params failed, disabling: {exc}", flush=True)
        _TRACK["on"] = False


def log_metric(key, value, step=None):
    if not _TRACK["on"]:
        return
    try:
        import mlflow

        mlflow.log_metric(key, float(value), step=step)
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] log_metric failed, disabling: {exc}", flush=True)
        _TRACK["on"] = False


def lsa_features(train_text, test_text):
    """One word 1-2-gram TF-IDF, one randomized SVD, L2-normalized dense factors."""
    all_text = pd.concat([train_text, test_text])
    vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="word",
                          token_pattern=r"\w{1,}", ngram_range=(1, 2),
                          min_df=3, max_df=0.9, max_features=100_000)
    vec.fit(all_text)
    x_all = vec.transform(all_text)
    print(f"  tfidf matrix {x_all.shape}", flush=True)

    svd = TruncatedSVD(n_components=N_COMPONENTS, algorithm="randomized",
                       n_iter=5, random_state=SEED)
    svd.fit(x_all)
    norm = Normalizer(copy=False)
    n_tr = len(train_text)
    x_tr = norm.fit_transform(svd.transform(x_all[:n_tr]))
    x_te = norm.fit_transform(svd.transform(x_all[n_tr:]))
    return x_tr, x_te


def fit_predict(x_tr, x_te, y):
    """One dense logistic regression per label; probabilities go straight out."""
    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [lsa] {lab} done", flush=True)
    return preds


def main():
    t0 = time.time()
    log_params({
        "approach": "lsa_tfidf_svd_logreg",
        "seed": SEED,
        "max_features": 100_000,
        "ngram": "1-2",
        "min_df": 3,
        "max_df": 0.9,
        "sublinear_tf": True,
        "svd_n_components": N_COMPONENTS,
        "svd_algorithm": "randomized",
        "svd_n_iter": 5,
        "normalizer": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    x_tr, x_te = lsa_features(train_text, test_text)
    tf = time.time() - t0
    print(f"features done at {tf:.0f}s", flush=True)
    log_metric("feature_seconds", tf, step=0)

    preds = fit_predict(x_tr, x_te, y)
    tm = time.time() - t0
    print(f"models done at {tm:.0f}s", flush=True)
    log_metric("model_seconds", tm - tf, step=0)

    sub = pd.DataFrame(preds, columns=LABELS)
    sub.insert(0, "id", test["id"].values)
    # emit rows in sample_submission order, ids as strings (leading-zero hex ids)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert sub[LABELS].notna().all().all(), "missing predictions for some sample ids"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    print(f"submission written at {time.time() - t0:.0f}s, shape={sub.shape}", flush=True)

    # graded by the hill's evaluator on submission/; no in-run grading
    total = time.time() - t0
    log_metric("total_seconds", total, step=0)
    print(f"total wall-clock {total:.0f}s")


if __name__ == "__main__":
    with _start_run():
        main()
