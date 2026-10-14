"""Complement Naive Bayes for jigsaw-toxic-comment-classification-challenge.

Rank-average of two text branches, with ComplementNB fit per label:
  (A) word 1-gram (10k) + char 2-6-gram (50k) TF-IDF
  (B) word 1-2-gram TF-IDF (200k), without signed feature reweighting

Both branches retain the baseline vectorizers fitted on train+test text.
Each classifier uses alpha=1.0 and norm=False.

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
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

COMP = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 0

# ---------------------------------------------------------------- MLflow (best effort)
_TRACK = {"on": True}


@contextlib.contextmanager
def _start_run():
    """Attach to the preconfigured MLflow run; never fail the training run."""
    stack = contextlib.ExitStack()
    try:
        import mlflow

        stack.enter_context(mlflow.start_run())
    except Exception as exc:  # pragma: no cover - telemetry only
        print(f"[mlflow] disabled: {exc}", flush=True)
        _TRACK["on"] = False
    try:
        yield
    finally:
        try:
            stack.close()
        except Exception as exc:  # pragma: no cover - telemetry only
            if _TRACK["on"]:
                print(f"[mlflow] shutdown failed, disabling: {exc}", flush=True)
            _TRACK["on"] = False


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


def branch_a(train_text, test_text, y):
    """word + char TF-IDF -> per-label Complement Naive Bayes."""
    all_text = pd.concat([train_text, test_text])
    word = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="word",
                           token_pattern=r"\w{1,}", stop_words="english",
                           ngram_range=(1, 1), max_features=10_000)
    char = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="char",
                           ngram_range=(2, 6), max_features=50_000)
    word.fit(all_text)
    char.fit(all_text)
    x_tr = hstack([word.transform(train_text), char.transform(train_text)]).tocsr()
    x_te = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = ComplementNB(alpha=1.0, norm=False)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds


def branch_b(train_text, test_text, y):
    """Complement Naive Bayes on nonnegative word 1-2 gram TF-IDF."""
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
        clf = ComplementNB(alpha=1.0, norm=False)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [B] {lab} done", flush=True)
    return preds


def to_rank(p):
    """Column-wise rank scaled to (0, 1) — AUC only cares about ordering."""
    out = np.zeros_like(p)
    for j in range(p.shape[1]):
        out[:, j] = rankdata(p[:, j]) / (p.shape[0] + 1.0)
    return out


def main():
    t0 = time.time()
    log_params({
        "approach": "rank_avg_word_char_tfidf_complement_nb + word_tfidf_complement_nb",
        "seed": SEED,
        "a_word_max_features": 10_000,
        "a_char_max_features": 50_000,
        "a_char_ngram": "2-6",
        "a_classifier": "ComplementNB",
        "a_alpha": 1.0,
        "a_norm": False,
        "b_max_features": 200_000,
        "b_ngram": "1-2",
        "b_min_df": 3,
        "b_max_df": 0.9,
        "b_classifier": "ComplementNB",
        "b_alpha": 1.0,
        "b_norm": False,
        "blend_weights": "0.5/0.5 rank average",
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

    blend = 0.5 * to_rank(pa) + 0.5 * to_rank(pb)

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
