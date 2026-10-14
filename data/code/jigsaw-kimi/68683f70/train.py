"""Word2Vec embedding baseline for jigsaw-toxic-comment-classification-challenge.

Single dense semantic model, replacing the baseline's two sparse TF-IDF
linear branches and their per-label rank blend:
  Word2Vec (CBOW, dim 200) trained on the competition corpus (90% sub-train
  + test text, transductive, mirroring the baseline's train+test vectorizer
  fits), average-pooled per comment -> one LogisticRegression per label.

Dense embeddings learned from the corpus capture distributional semantics, so
the classifier can transfer evidence between semantically similar comments
(synonyms, paraphrases, misspellings of slurs) instead of relying on exact
n-gram matches.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import re
import time

import numpy as np
import pandas as pd
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


_TOKEN_RE = re.compile(r"\w+")


def tokenize(texts):
    """Lowercase \\w+ tokenization for every comment."""
    return [_TOKEN_RE.findall(t.lower()) for t in texts]


def train_w2v(sentences):
    """Word2Vec on the given tokenized sentences (transductive: sub-train + test)."""
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=sentences,
        vector_size=200,
        window=5,
        min_count=5,
        sg=0,
        negative=5,
        epochs=5,
        workers=4,
        seed=SEED,
    )
    return model


def pool(sentences, wv):
    """Mean of in-vocabulary Word2Vec vectors per comment (zeros if none)."""
    key_to_idx = wv.key_to_index
    vectors = wv.vectors  # float32, (vocab, dim)
    dim = vectors.shape[1]
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, toks in enumerate(sentences):
        idxs = [key_to_idx[t] for t in toks if t in key_to_idx]
        if idxs:
            out[i] = vectors[idxs].mean(axis=0)
    return out


def main():
    t0 = time.time()
    log_params({
        "approach": "corpus_word2vec_cbow_avgpool + per-label logreg",
        "seed": SEED,
        "w2v_vector_size": 200,
        "w2v_window": 5,
        "w2v_min_count": 5,
        "w2v_sg": 0,
        "w2v_negative": 5,
        "w2v_epochs": 5,
        "w2v_workers": 4,
        "w2v_corpus": "sub-train + test (transductive)",
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

    # 90/10 holdout for internal validation; models are trained once on the 90%.
    idx_tr, idx_ho = train_test_split(
        np.arange(len(train)), test_size=0.1, random_state=SEED,
        stratify=train["toxic"],
    )
    sub_text = train_text.iloc[idx_tr]
    ho_text = train_text.iloc[idx_ho]
    y_sub = y[idx_tr]
    y_ho = y[idx_ho]
    print(f"sub-train {len(idx_tr)} rows, holdout {len(idx_ho)} rows", flush=True)

    # tokenize everything once
    sub_tok = tokenize(sub_text)
    ho_tok = tokenize(ho_text)
    te_tok = tokenize(test_text)
    print(f"tokenized at {time.time() - t0:.0f}s", flush=True)

    # transductive embedding training: sub-train + test text (mirrors the
    # baseline's practice of fitting vectorizers on train+test text)
    w2v = train_w2v(sub_tok + te_tok)
    tw = time.time() - t0
    print(f"word2vec trained at {tw:.0f}s, vocab={len(w2v.wv)}", flush=True)
    log_metric("word2vec_seconds", tw, step=0)
    log_params({"w2v_vocab": len(w2v.wv)})

    X_tr = pool(sub_tok, w2v.wv)
    X_ho = pool(ho_tok, w2v.wv)
    X_te = pool(te_tok, w2v.wv)
    print(f"pooled at {time.time() - t0:.0f}s, X_tr={X_tr.shape}", flush=True)

    preds = np.zeros((X_te.shape[0], len(LABELS)))
    preds_ho = np.zeros((X_ho.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_sub[:, j])
        preds[:, j] = clf.predict_proba(X_te)[:, 1]
        preds_ho[:, j] = clf.predict_proba(X_ho)[:, 1]
        auc = roc_auc_score(y_ho[:, j], preds_ho[:, j])
        print(f"  [w2v-lr] {lab} done, holdout_auc={auc:.5f}", flush=True)
        log_metric(f"holdout_auc_{lab}", auc, step=0)

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
