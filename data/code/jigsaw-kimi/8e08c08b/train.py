"""Baseline for jigsaw-toxic-comment-classification-challenge.

Rank-average of two classic linear text branches, fit per label:
  (A) word 1-gram (10k) + char_wb 2-6-gram (50k) TF-IDF -> LogisticRegression C=0.1, solver=sag
  (B) NB-LogReg (Naive-Bayes-weighted features) on word 1-2-gram TF-IDF -> LogisticRegression C=4

Both vectorizers are fit on train+test text, as the source kernels do.
Deliberately a solid mid-competition solution: linear models on sparse
n-grams, no neural nets, no pretrained embeddings.

This experiment replaces the fixed 0.5/0.5 rank blend with per-label blend
weights tuned for holdout AUC on a 10% stratified split of train; both
branches are fit once on the 90% sub-train (no full-data refit).

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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


def branch_a(train_text, test_text, y, holdout_text):
    """word + char TF-IDF -> per-label logistic regression.

    Returns (test_preds, holdout_preds) from one fit on train_text.
    """
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
    x_ho = hstack([word.transform(holdout_text), char.transform(holdout_text)]).tocsr()

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    preds_ho = np.zeros((x_ho.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=0.1, solver="sag", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        preds_ho[:, j] = clf.predict_proba(x_ho)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds, preds_ho


def branch_b(train_text, test_text, y, holdout_text):
    """Naive-Bayes-weighted features (NB-LogReg) on word 1-2 gram TF-IDF.

    Returns (test_preds, holdout_preds) from one fit on train_text.
    """
    all_text = pd.concat([train_text, test_text])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                          strip_accents="unicode", token_pattern=r"\w{1,}",
                          sublinear_tf=True, smooth_idf=True, use_idf=True,
                          max_features=200_000)
    vec.fit(all_text)
    x_tr = vec.transform(train_text)
    x_te = vec.transform(test_text)
    x_ho = vec.transform(holdout_text)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    preds_ho = np.zeros((x_ho.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        yj = y[:, j]
        p = (1.0 + np.asarray(x_tr[yj == 1].sum(axis=0))) / (1.0 + (yj == 1).sum())
        q = (1.0 + np.asarray(x_tr[yj == 0].sum(axis=0))) / (1.0 + (yj == 0).sum())
        r = np.log(p / q).ravel()
        clf = LogisticRegression(C=4.0, dual=True, solver="liblinear", random_state=SEED)
        clf.fit(x_tr.multiply(r).tocsr(), yj)
        preds[:, j] = clf.predict_proba(x_te.multiply(r).tocsr())[:, 1]
        preds_ho[:, j] = clf.predict_proba(x_ho.multiply(r).tocsr())[:, 1]
        print(f"  [B] {lab} done", flush=True)
    return preds, preds_ho


def branch_c(train_text, test_text, y, holdout_text):
    """Naive-Bayes-weighted features (NB-LogReg) on char_wb 3-5 gram TF-IDF.

    Same structure as branch_b but character n-grams, so it catches obfuscated
    spellings the word branches miss. Returns (test_preds, holdout_preds).
    """
    all_text = pd.concat([train_text, test_text])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                          max_df=0.9, max_features=100_000, sublinear_tf=True,
                          smooth_idf=True, use_idf=True, strip_accents="unicode")
    vec.fit(all_text)
    x_tr = vec.transform(train_text)
    x_te = vec.transform(test_text)
    x_ho = vec.transform(holdout_text)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    preds_ho = np.zeros((x_ho.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        yj = y[:, j]
        p = (1.0 + np.asarray(x_tr[yj == 1].sum(axis=0))) / (1.0 + (yj == 1).sum())
        q = (1.0 + np.asarray(x_tr[yj == 0].sum(axis=0))) / (1.0 + (yj == 0).sum())
        r = np.log(p / q).ravel()
        clf = LogisticRegression(C=4.0, dual=True, solver="liblinear", random_state=SEED)
        clf.fit(x_tr.multiply(r).tocsr(), yj)
        preds[:, j] = clf.predict_proba(x_te.multiply(r).tocsr())[:, 1]
        preds_ho[:, j] = clf.predict_proba(x_ho.multiply(r).tocsr())[:, 1]
        print(f"  [C] {lab} done", flush=True)
    return preds, preds_ho


def to_rank(p):
    """Column-wise rank scaled to (0, 1) — AUC only cares about ordering."""
    out = np.zeros_like(p)
    for j in range(p.shape[1]):
        out[:, j] = rankdata(p[:, j]) / (p.shape[0] + 1.0)
    return out


def main():
    t0 = time.time()
    log_params({
        "approach": "rank_avg word+char logreg + word nb_logreg + char nb_logreg",
        "c_max_features": 100_000,
        "c_ngram": "3-5",
        "c_analyzer": "char_wb",
        "c_C": 4.0,
        "c_solver": "liblinear",
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
        "blend_weights": "per-label 3-way simplex, tuned on 10% holdout (step 0.1)",
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    # 90/10 holdout for tuning the per-label A/B blend weights; models are
    # trained once on the 90% and NOT refit on full data (time budget).
    idx_tr, idx_ho = train_test_split(
        np.arange(len(train)), test_size=0.1, random_state=SEED,
        stratify=train["toxic"],
    )
    sub_text = train_text.iloc[idx_tr]
    ho_text = train_text.iloc[idx_ho]
    y_sub = y[idx_tr]
    y_ho = y[idx_ho]
    print(f"sub-train {len(idx_tr)} rows, holdout {len(idx_ho)} rows", flush=True)

    pa, pa_ho = branch_a(sub_text, test_text, y_sub, ho_text)
    ta = time.time() - t0
    print(f"branch A done at {ta:.0f}s", flush=True)
    log_metric("branch_a_seconds", ta, step=0)
    pb, pb_ho = branch_b(sub_text, test_text, y_sub, ho_text)
    tb = time.time() - t0
    print(f"branch B done at {tb:.0f}s", flush=True)
    log_metric("branch_b_seconds", tb - ta, step=0)
    pc, pc_ho = branch_c(sub_text, test_text, y_sub, ho_text)
    tc = time.time() - t0
    print(f"branch C done at {tc:.0f}s", flush=True)
    log_metric("branch_c_seconds", tc - tb, step=0)

    rank_a_ho = to_rank(pa_ho)
    rank_b_ho = to_rank(pb_ho)
    rank_c_ho = to_rank(pc_ho)
    rank_a = to_rank(pa)
    rank_b = to_rank(pb)
    rank_c = to_rank(pc)

    # per-label simplex search over the three branch weights (holdout AUC)
    weights = np.zeros((len(LABELS), 3))
    blend = np.zeros_like(rank_a)
    for j, lab in enumerate(LABELS):
        best, best_auc = (1.0 / 3, 1.0 / 3, 1.0 / 3), -1.0
        for w_a in np.arange(0.0, 1.01, 0.1):
            for w_b in np.arange(0.0, 1.01 - w_a, 0.1):
                w_c = 1.0 - w_a - w_b
                auc = roc_auc_score(
                    y_ho[:, j],
                    w_a * rank_a_ho[:, j] + w_b * rank_b_ho[:, j] + w_c * rank_c_ho[:, j],
                )
                if auc > best_auc:
                    best_auc, best = auc, (w_a, w_b, w_c)
        w_a, w_b, w_c = best
        weights[j] = best
        blend[:, j] = w_a * rank_a[:, j] + w_b * rank_b[:, j] + w_c * rank_c[:, j]
        print(f"  blend[{lab}] w_A={w_a:.2f} w_B={w_b:.2f} w_C={w_c:.2f} "
              f"holdout_auc={best_auc:.5f}", flush=True)
        log_metric(f"holdout_auc_{lab}", best_auc, step=0)
    log_params({f"blend_w_{br}_{lab}": float(weights[j][k])
                for j, lab in enumerate(LABELS)
                for k, br in enumerate(("A", "B", "C"))})

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
