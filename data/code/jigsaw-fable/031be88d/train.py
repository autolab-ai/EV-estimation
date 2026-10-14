"""Baseline for jigsaw-toxic-comment-classification-challenge.

Rank-average of two classic linear text branches, fit per label:
  (A) mean-pooled Word2Vec (200d, skip-gram, train+test text) -> LogisticRegression C=0.1, solver=lbfgs
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
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

COMP = "jigsaw-toxic-comment-classification-challenge"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 0
W2V_WORKERS = int(os.environ.get("W2V_WORKERS", "4"))

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


W2V = dict(vector_size=200, window=5, min_count=2, sg=1, negative=5, epochs=5)


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def branch_a(train_text, test_text, y):
    """mean-pooled Word2Vec comment embeddings (trained on train+test text) -> per-label logistic regression."""
    from gensim.models import Word2Vec
    tok_tr = [_tokenize(t) for t in train_text]
    tok_te = [_tokenize(t) for t in test_text]
    t0 = time.time()
    w2v = Word2Vec(sentences=tok_tr + tok_te, workers=W2V_WORKERS, seed=SEED, **W2V)
    print(f"  [A] word2vec trained in {time.time() - t0:.0f}s, vocab={len(w2v.wv)}", flush=True)
    kv = w2v.wv

    def embed(token_lists):
        out = np.zeros((len(token_lists), W2V["vector_size"]), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            idx = [kv.key_to_index[t] for t in toks if t in kv.key_to_index]
            if idx:
                out[i] = kv.vectors[idx].mean(axis=0)
        return out

    e_tr, e_te = embed(tok_tr), embed(tok_te)
    scaler = StandardScaler().fit(np.vstack([e_tr, e_te]))
    x_tr = scaler.transform(e_tr).astype(np.float32)
    x_te = scaler.transform(e_te).astype(np.float32)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        # lbfgs instead of sag: sag needs ~1000 full passes on dense standardized inputs
        # (a sibling run timed out on it); lbfgs reaches the same optimum in seconds.
        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds, x_tr, x_te


def branch_mlp(x_tr, x_te, y):
    """Multi-label MLP (one shared 64-unit ReLU layer, six sigmoid outputs) on branch A's dense Word2Vec features."""
    Y = y.astype(int)
    clf = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                        max_iter=3, shuffle=True, random_state=0, early_stopping=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf.fit(x_tr, Y)
    p = clf.predict_proba(x_te)
    assert p.shape == (x_te.shape[0], len(LABELS))
    print("  [MLP] done", flush=True)
    return p


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
        yj = y[:, j]
        p = (1.0 + np.asarray(x_tr[yj == 1].sum(axis=0))) / (1.0 + (yj == 1).sum())
        q = (1.0 + np.asarray(x_tr[yj == 0].sum(axis=0))) / (1.0 + (yj == 0).sum())
        r = np.log(p / q).ravel()
        clf = LogisticRegression(C=4.0, dual=True, solver="liblinear", random_state=SEED)
        clf.fit(x_tr.multiply(r).tocsr(), yj)
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
        "approach": "rank_avg_w2v_meanpool_logreg + nb_logreg + mlp",
        "seed": SEED,
        "a_features": "word2vec_mean_pool",
        "a_w2v_vector_size": W2V["vector_size"],
        "a_w2v_window": W2V["window"],
        "a_w2v_min_count": W2V["min_count"],
        "a_w2v_sg": W2V["sg"],
        "a_w2v_epochs": W2V["epochs"],
        "a_C": 0.1,
        "a_solver": "lbfgs",
        "b_max_features": 200_000,
        "b_ngram": "1-2",
        "b_min_df": 3,
        "b_max_df": 0.9,
        "b_C": 4.0,
        "b_solver": "liblinear",
        "blend_weights": "1/3 rank average of A, B, MLP",
        "mlp_hidden": 64,
        "mlp_max_iter": 3,
        "mlp_batch_size": 256,
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    pa, x_tr, x_te = branch_a(train_text, test_text, y)
    ta = time.time() - t0
    print(f"branch A done at {ta:.0f}s", flush=True)
    log_metric("branch_a_seconds", ta, step=0)
    pb = branch_b(train_text, test_text, y)
    tb = time.time() - t0
    print(f"branch B done at {tb:.0f}s", flush=True)
    log_metric("branch_b_seconds", tb - ta, step=0)
    p_mlp = branch_mlp(x_tr, x_te, y)
    tm = time.time() - t0
    print(f"branch MLP done at {tm:.0f}s", flush=True)
    log_metric("branch_mlp_seconds", tm - tb, step=0)

    blend = (to_rank(pa) + to_rank(pb) + to_rank(p_mlp)) / 3.0

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
