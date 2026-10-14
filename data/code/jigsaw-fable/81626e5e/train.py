"""Baseline for jigsaw-toxic-comment-classification-challenge.

Rank-average of two classic linear text branches, fit per label:
  (A) word 1-gram (10k) + char_wb 2-6-gram (50k) TF-IDF -> LogisticRegression C=0.1, solver=sag
  (B) NB-LogReg (Naive-Bayes-weighted features) on word 1-2-gram TF-IDF -> LogisticRegression C=4

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
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

COMP = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 0

# ---------------------------------------------------------------- text normalization
_RE_WS_CHARS = re.compile(r"[\n\r\t]")
_RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_TIMESTAMP = re.compile(r"\d{1,2}:\d{2}, \d{1,2} [a-z]+ \d{4} \(utc\)")
_RE_REPEAT = re.compile(r"(.)\1{2,}")
_DEOBFUSCATE = [(re.compile(p), f" {r} ") for p, r in [
    (r"\bf+\W*u+\W*c+\W*k+\b", "fuck"),
    (r"\bf[\W_]*[*@#$%&]+[\W_]*c?k\b", "fuck"),
    (r"\b(?:fuk|fcuk|phuck|fvck|fuq|fux)\b", "fuck"),
    (r"\bs\W*h\W*[i1!|]\W*t+\b", "shit"),
    (r"\bb\W*[i1!|*]\W*t\W*c\W*h(?:es)?\b", "bitch"),
    (r"\ba\W*s\W*s\W*h\W*[o0]\W*l\W*es?\b", "asshole"),
    (r"(?<!\w)(?:@ss|a\$\$|a55)\b", "ass"),
    (r"\bd[i1!]cks?\b", "dick"),
    (r"\bc\W*u\W*n\W*ts?\b", "cunt"),
    (r"\bpu\W*[s$5]\W*[s$5]\W*y\b", "pussy"),
    (r"\bn[i1!]gg+[ae3@]+r?s?\b", "nigger"),
    (r"\bf[a@]gg?[o0i1e]ts?\b", "faggot"),
    (r"\bwh[o0]res?\b", "whore"),
    (r"(?<!\w)[s$5]u+ck(?:s|ed|er|ers|ing)?\b", "suck"),
    (r"\bstfu\b", "shut the fuck up"),
    (r"\bwtf\b", "what the fuck"),
]]
_SHORTHAND = [(re.compile(p), r) for p, r in [
    (r"\bu\b", " you "),
    (r"\bur\b", " your "),
    (r"\br\b", " are "),
]]
_RE_SPACES = re.compile(r"\s+")


def normalize(t):
    """Lowercase, strip noise tokens, collapse repeats, de-obfuscate profanity."""
    t = t.lower()
    t = _RE_WS_CHARS.sub(" ", t)
    t = _RE_IP.sub(" ipaddress ", t)
    t = _RE_URL.sub(" url ", t)
    t = _RE_TIMESTAMP.sub("", t)
    t = _RE_REPEAT.sub(r"\1\1", t)
    for rx, rep in _DEOBFUSCATE:
        t = rx.sub(rep, t)
    for rx, rep in _SHORTHAND:
        t = rx.sub(rep, t)
    return _RE_SPACES.sub(" ", t).strip()


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


def balanced_index(yj, seed):
    """Indices of all positives plus an equal-size random subset of negatives."""
    rng = np.random.RandomState(seed)
    pos = np.flatnonzero(yj == 1)
    neg = np.flatnonzero(yj == 0)
    neg_keep = rng.choice(neg, size=len(pos), replace=False)
    return np.sort(np.concatenate([pos, neg_keep]))


def branch_a(train_text, test_text, y):
    """word + char TF-IDF -> per-label logistic regression."""
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

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        idx = balanced_index(y[:, j], SEED + j)
        log_metric(f"train_rows_{lab}", len(idx), step=0)
        clf = LogisticRegression(C=0.1, solver="sag", max_iter=1000, random_state=SEED)
        clf.fit(x_tr[idx], y[idx, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds


def branch_b(train_text, test_text, y):
    """Naive-Bayes-weighted features (NB-LogReg) on word 1-2 gram TF-IDF."""
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
        idx = balanced_index(y[:, j], SEED + j)
        yj = y[idx, j]
        x_bal = x_tr[idx]
        p = (1.0 + np.asarray(x_bal[yj == 1].sum(axis=0))) / (1.0 + (yj == 1).sum())
        q = (1.0 + np.asarray(x_bal[yj == 0].sum(axis=0))) / (1.0 + (yj == 0).sum())
        r = np.log(p / q).ravel()
        clf = LogisticRegression(C=4.0, dual=True, solver="liblinear", random_state=SEED)
        clf.fit(x_bal.multiply(r).tocsr(), yj)
        preds[:, j] = clf.predict_proba(x_te.multiply(r).tocsr())[:, 1]
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
        "approach": "rank_avg_word_char_tfidf_logreg + nb_logreg",
        "seed": SEED,
        "undersample": "balanced_1to1_per_label",
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

    train_text = train["comment_text"].fillna("unknown").map(normalize)
    test_text = test["comment_text"].fillna("unknown").map(normalize)
    print(f"text normalized at {time.time() - t0:.0f}s", flush=True)
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
