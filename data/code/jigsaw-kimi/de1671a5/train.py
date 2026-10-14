"""Dense-representation baseline for jigsaw-toxic-comment-classification-challenge.

One pipeline, no sparse n-gram branches and no blending:
  1. regex word tokenization of every comment (lowercased \\w+ tokens)
  2. skip-gram Word2Vec (200d) trained from scratch on train+test tokens
  3. each comment = unweighted mean of its in-vocabulary token vectors
  4. one LogisticRegression per label on the 200d document vectors

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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



W2V = dict(sg=1, vector_size=200, window=5, min_count=5, negative=10, hs=0, epochs=5,
           workers=int(os.environ.get("W2V_WORKERS", "4")), seed=SEED)

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def mean_pool(token_lists, kv, dim, chunk=20_000):
    """Mean of in-vocabulary token vectors per document.

    Vectorized with np.add.at over flat (row, vocab_index) arrays, in document
    chunks so the gathered vector block never gets large.
    """
    key_to_index = kv.key_to_index
    vectors = kv.vectors
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for start in range(0, len(token_lists), chunk):
        block = token_lists[start:start + chunk]
        rows = []
        cols = []
        for i, toks in enumerate(block):
            for t in toks:
                j = key_to_index.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        if not rows:
            continue
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        acc = np.zeros((len(block), dim), dtype=np.float32)
        np.add.at(acc, rows, vectors[cols])
        counts = np.bincount(rows, minlength=len(block)).astype(np.float32)
        nz = counts > 0
        acc[nz] /= counts[nz, None]
        out[start:start + len(block)] = acc
    return out


def main():
    t0 = time.time()
    log_params({
        "approach": "corpus_word2vec_meanpool_logreg",
        "seed": SEED,
        "tokenizer": "regex_lower_w+",
        "w2v_sg": W2V["sg"],
        "w2v_vector_size": W2V["vector_size"],
        "w2v_window": W2V["window"],
        "w2v_min_count": W2V["min_count"],
        "w2v_negative": W2V["negative"],
        "w2v_epochs": W2V["epochs"],
        "w2v_trained_on": "train+test tokens",
        "pooling": "unweighted_mean_in_vocab",
        "clf": "LogisticRegression",
        "clf_C": 1.0,
        "clf_solver": "lbfgs",
        "clf_max_iter": 1000,
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    tr_tokens = [tokenize(t) for t in train_text]
    te_tokens = [tokenize(t) for t in test_text]
    t_tok = time.time() - t0
    print(f"tokenized at {t_tok:.0f}s", flush=True)
    log_metric("tokenize_seconds", t_tok, step=0)

    from gensim.models import Word2Vec

    model = Word2Vec(sentences=tr_tokens + te_tokens, **W2V)
    kv = model.wv
    t_w2v = time.time() - t0
    print(f"word2vec trained at {t_w2v:.0f}s, vocab={len(kv.key_to_index)}", flush=True)
    log_metric("w2v_seconds", t_w2v - t_tok, step=0)
    log_metric("w2v_vocab", len(kv.key_to_index), step=0)

    dim = W2V["vector_size"]
    x_tr = mean_pool(tr_tokens, kv, dim)
    x_te = mean_pool(te_tokens, kv, dim)
    t_pool = time.time() - t0
    print(f"pooled at {t_pool:.0f}s", flush=True)
    log_metric("pool_seconds", t_pool - t_w2v, step=0)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  {lab} done at {time.time() - t0:.0f}s", flush=True)
    log_metric("clf_seconds", time.time() - t0 - t_pool, step=0)

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
