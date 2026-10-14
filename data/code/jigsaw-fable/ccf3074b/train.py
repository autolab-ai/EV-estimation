"""Baseline for jigsaw-toxic-comment-classification-challenge.

Rank-average of two classic linear text branches, fit per label:
  (A) word 1-gram (10k) + char_wb 2-6-gram (50k) TF-IDF -> LogisticRegression C=0.1, solver=sag
  (B) MultinomialNB(alpha=1.0) on word 1-2-gram TF-IDF

Both vectorizers are fit on train+test text, as the source kernels do.
Deliberately a solid mid-competition solution: linear models on sparse
n-grams, no neural nets, no pretrained embeddings, no tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

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


def branch_a(train_text, test_text, y, z_train=None, z_test=None):
    """word + char TF-IDF (+ optional dense stacked columns) -> per-label logistic regression."""
    all_text = pd.concat([train_text, test_text])
    word = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="word",
                           token_pattern=r"\w{1,}", stop_words="english",
                           ngram_range=(1, 1), max_features=10_000)
    char = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="char_wb",
                           ngram_range=(2, 6), max_features=50_000)
    word.fit(all_text)
    char.fit(all_text)
    tr_parts = [word.transform(train_text), char.transform(train_text)]
    te_parts = [word.transform(test_text), char.transform(test_text)]
    if z_train is not None:
        # feature-level stacking: append branch-B's OOF view of all six labels
        tr_parts.append(csr_matrix(z_train))
        te_parts.append(csr_matrix(z_test))
    x_tr = hstack(tr_parts).tocsr()
    x_te = hstack(te_parts).tocsr()

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=0.1, solver="sag", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds


def _nb_logreg_fit_predict(x_fit, y_fit, x_preds):
    # Generative MultinomialNB fit directly on the non-negative TF-IDF features
    # (replaces the NB-ratio-scaled LogisticRegression hybrid).
    clf = MultinomialNB(alpha=1.0)
    clf.fit(x_fit, y_fit)
    return [clf.predict_proba(x)[:, 1] for x in x_preds]


def branch_b(train_text, test_text, y, n_folds=5):
    """MultinomialNB on word 1-2 gram TF-IDF.

    Returns (full-train test preds, out-of-fold train preds, fold-averaged test preds).
    """
    all_text = pd.concat([train_text, test_text])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                          strip_accents="unicode", token_pattern=r"\w{1,}",
                          sublinear_tf=True, smooth_idf=True, use_idf=True,
                          max_features=200_000)
    vec.fit(all_text)
    x_tr = vec.transform(train_text).tocsr()
    x_te = vec.transform(test_text).tocsr()

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        preds[:, j] = _nb_logreg_fit_predict(x_tr, y[:, j], [x_te])[0]
        print(f"  [B] {lab} done", flush=True)

    # out-of-fold predictions (NB ratio computed from training-fold rows only)
    oof = np.zeros((x_tr.shape[0], len(LABELS)))
    test_fold = np.zeros((n_folds, x_te.shape[0], len(LABELS)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for k, (tr_idx, va_idx) in enumerate(skf.split(x_tr, y[:, 0])):
        x_fit, x_val = x_tr[tr_idx], x_tr[va_idx]
        for j in range(len(LABELS)):
            pv, pt = _nb_logreg_fit_predict(x_fit, y[tr_idx, j], [x_val, x_te])
            oof[va_idx, j] = pv
            test_fold[k, :, j] = pt
        print(f"  [B-OOF] fold {k + 1}/{n_folds} done", flush=True)
    return preds, oof, test_fold.mean(axis=0)


def stack_features(p):
    """Bounded logit of probabilities, scaled so the added row norm stays small (~<1)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1.0 - p)), -8.0, 8.0) / 20.0


def to_rank(p):
    """Column-wise rank scaled to (0, 1) — AUC only cares about ordering."""
    out = np.zeros_like(p)
    for j in range(p.shape[1]):
        out[:, j] = rankdata(p[:, j]) / (p.shape[0] + 1.0)
    return out


def main():
    t0 = time.time()
    log_params({
        "approach": "rank_avg_word_char_tfidf_logreg(+5fold OOF nb_logreg stack) + nb_logreg",
        "stack_oof_folds": 5,
        "stack_scale": 20.0,
        "seed": SEED,
        "a_word_max_features": 10_000,
        "a_char_max_features": 50_000,
        "a_char_ngram": "2-6",
        "a_C": 0.1,
        "a_solver": "sag",
        "b_max_features": 200_000,
        "b_ngram": "1-2",
        "b_min_df": 3,
        "b_max_df": 0.9,
        "b_C": 4.0,
        "b_solver": "liblinear",
        "blend_weights": "0.5/0.5 rank average",
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    pb, oof_b, test_b = branch_b(train_text, test_text, y)
    tb = time.time() - t0
    print(f"branch B (+OOF) done at {tb:.0f}s", flush=True)
    log_metric("branch_b_seconds", tb, step=0)
    log_metric("stage1_oof_seconds", tb, step=0)
    pa = branch_a(train_text, test_text, y, stack_features(oof_b), stack_features(test_b))
    ta = time.time() - t0
    print(f"branch A (stacked) done at {ta:.0f}s", flush=True)
    log_metric("branch_a_seconds", ta - tb, step=0)

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
