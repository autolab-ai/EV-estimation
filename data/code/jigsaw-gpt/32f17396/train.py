"""Rank-average of two text classifiers for toxic comments:
  (A) word + char_wb TF-IDF -> shared 32-component SVD -> per-label logistic regression
  (B) jointly trained multilabel fastText, without pretrained vectors

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

COMP = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 0
with open(os.environ.get("FASTTEXT_CONFIG", Path(__file__).with_name("fasttext_config.json"))) as handle:
    FASTTEXT_CONFIG = json.load(handle)

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


def branch_a(train_text, test_text, y):
    """word + char TF-IDF -> shared latent space -> per-label logistic regression."""
    all_text = pd.concat([train_text, test_text])
    word = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="word",
                           token_pattern=r"\w{1,}", stop_words="english",
                           ngram_range=(1, 1), max_features=10_000, dtype=np.float32)
    char = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="char_wb",
                           ngram_range=(2, 6), max_features=50_000, dtype=np.float32)
    word.fit(all_text)
    char.fit(all_text)
    x_tr = hstack([word.transform(train_text), char.transform(train_text)]).tocsr()
    x_te = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()

    svd = TruncatedSVD(n_components=32, algorithm="randomized", n_iter=2,
                       n_oversamples=10, random_state=SEED)
    with threadpool_limits(limits=4, user_api="blas"):
        x_tr = svd.fit_transform(x_tr).astype(np.float32, copy=False)
        x_te = svd.transform(x_te).astype(np.float32, copy=False)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=0.1, solver="sag", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds


def fasttext_text(text):
    """Escape tokens so comment content cannot become fastText labels."""
    text = "unknown" if pd.isna(text) else str(text)
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())[:256]
    return " ".join("w_" + token for token in tokens) if tokens else "w_empty"


def branch_b(train_text, test_text, y):
    """One joint multilabel fastText model with independent label outputs."""
    with tempfile.TemporaryDirectory(prefix="jigsaw_fasttext_") as tmp:
        training_file = os.path.join(tmp, "train.txt")
        with open(training_file, "w", encoding="utf-8") as handle:
            for text, targets in zip(train_text, y):
                labels = ["__label__" + lab for lab, value in zip(LABELS, targets) if value == 1]
                if not labels:
                    labels = ["__label__clean"]
                handle.write(" ".join(labels) + " " + fasttext_text(text) + "\n")
        model = fasttext.train_supervised(input=training_file, **FASTTEXT_CONFIG["training"])

    preds = np.zeros((len(test_text), len(LABELS)))
    label_columns = {"__label__" + lab: j for j, lab in enumerate(LABELS)}
    batch_size = FASTTEXT_CONFIG["prediction_batch_size"]
    for start in range(0, len(test_text), batch_size):
        batch = [fasttext_text(text) for text in test_text.iloc[start:start + batch_size]]
        batch_labels, batch_probs = model.predict(batch, k=-1)
        for row, (labels, probs) in enumerate(zip(batch_labels, batch_probs), start=start):
            for label, probability in zip(labels, probs):
                if label in label_columns:
                    preds[row, label_columns[label]] = probability
    np.clip(preds, 0.0, 1.0, out=preds)
    print("  [B] joint multilabel fastText done", flush=True)
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
        "approach": "rank_avg_word_char_tfidf_svd_logreg + fasttext",
        "seed": SEED,
        "a_word_max_features": 10_000,
        "a_char_max_features": 50_000,
        "a_char_ngram": "2-6",
        "a_C": 0.1,
        "a_solver": "sag",
        "a_svd_components": 32,
        "a_svd_algorithm": "randomized",
        "a_svd_n_iter": 2,
        "a_svd_n_oversamples": 10,
        "a_svd_blas_threads": 4,
        "a_dtype": "float32",
        **{"b_" + key: value for key, value in FASTTEXT_CONFIG["training"].items()},
        "b_prediction_batch_size": FASTTEXT_CONFIG["prediction_batch_size"],
        "b_max_tokens": 256,
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
