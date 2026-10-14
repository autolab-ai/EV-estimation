"""Cosine class-centroid classifiers for toxic comment classification.

Logit-average of two sparse text branches, fit per label:
  (A) word 1-gram (10k) + char_wb 2-6-gram (50k) TF-IDF
  (B) Naive-Bayes-weighted word 1-2-gram TF-IDF

Both branches L2-normalize document rows, average each class's rows,
L2-normalize those centroids, and apply expit to the difference in cosine
similarity. Vectorizers remain fit on train+test text.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

COMP = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 0

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


def cosine_centroid_probabilities(x_tr, x_te, y):
    """Predict from sparse, L2-normalized rows using normalized class means."""
    centroids = []
    for label in (1, 0):
        mask = y == label
        centroid = np.asarray(x_tr[mask].sum(axis=0)).ravel()
        count = np.count_nonzero(mask)
        if count:
            centroid /= count
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroids.append(centroid)
    return expit(np.asarray(x_te @ (centroids[0] - centroids[1])).ravel())


def branch_a(train_text, test_text, y):
    """word + char TF-IDF -> per-label cosine class-centroid classifier."""
    all_text = pd.concat([train_text, test_text])
    word = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="word",
                           token_pattern=r"\w{1,}", stop_words="english",
                           ngram_range=(1, 1), max_features=10_000)
    char = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="char_wb",
                           ngram_range=(2, 6), max_features=50_000)
    word.fit(all_text)
    char.fit(all_text)
    x_tr = hstack([word.transform(train_text), char.transform(train_text)]).tocsr()
    x_te = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()

    x_tr = normalize(x_tr, norm="l2", copy=False)
    x_te = normalize(x_te, norm="l2", copy=False)
    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        preds[:, j] = cosine_centroid_probabilities(x_tr, x_te, y[:, j])
        print(f"  [A] {lab} done", flush=True)
    return preds


def branch_b(train_text, test_text, y):
    """Naive-Bayes-weighted word 1-2 gram TF-IDF -> cosine class centroids."""
    all_text = pd.concat([train_text, test_text])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                          strip_accents="unicode", token_pattern=r"\w{1,}",
                          sublinear_tf=True, smooth_idf=True, use_idf=True,
                          max_features=200_000)
    vec.fit(all_text)
    x_tr = vec.transform(train_text)
    x_te = vec.transform(test_text)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        yj = y[:, j]
        p = (1.0 + np.asarray(x_tr[yj == 1].sum(axis=0))) / (1.0 + (yj == 1).sum())
        q = (1.0 + np.asarray(x_tr[yj == 0].sum(axis=0))) / (1.0 + (yj == 0).sum())
        r = np.log(p / q).ravel()
        weighted_tr = normalize(x_tr.multiply(r).tocsr(), norm="l2", copy=False)
        weighted_te = normalize(x_te.multiply(r).tocsr(), norm="l2", copy=False)
        preds[:, j] = cosine_centroid_probabilities(weighted_tr, weighted_te, yj)
        del weighted_tr, weighted_te
        print(f"  [B] {lab} done", flush=True)
    return preds


def main():
    t0 = time.time()
    log_params({
        "approach": "logit_avg_word_char_tfidf_centroid + nb_centroid",
        "seed": SEED,
        "a_word_max_features": 10_000,
        "a_char_max_features": 50_000,
        "a_char_ngram": "2-6",
        "a_classifier": "cosine_class_centroid",
        "b_max_features": 200_000,
        "b_ngram": "1-2",
        "b_min_df": 3,
        "b_max_df": 0.9,
        "b_classifier": "cosine_class_centroid",
        "document_norm": "l2",
        "centroid_norm": "l2",
        "blend_weights": "0.5/0.5 logit average",
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    pa = branch_a(train_text, test_text, y)
    ta = time.time() - t0
    print(f"branch A done at {ta:.0f}s", flush=True)
    log_metric("branch_a_seconds", ta, step=0)
    pb = branch_b(train_text, test_text, y)
    tb = time.time() - t0
    print(f"branch B done at {tb:.0f}s", flush=True)
    log_metric("branch_b_seconds", tb - ta, step=0)

    pa_clipped = np.clip(pa, 1e-12, 1 - 1e-12)
    pb_clipped = np.clip(pb, 1e-12, 1 - 1e-12)
    blend = expit(0.5 * logit(pa_clipped) + 0.5 * logit(pb_clipped))

    sub = pd.DataFrame(blend, columns=LABELS)
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
