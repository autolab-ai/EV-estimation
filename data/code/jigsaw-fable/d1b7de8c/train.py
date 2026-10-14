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
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
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


def build_features(train_text, test_text):
    """Fit the three TF-IDF vectorizers once on train+test text; return CSR blocks."""
    all_text = pd.concat([train_text, test_text])
    a_word = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="word",
                             token_pattern=r"\w{1,}", stop_words="english",
                             ngram_range=(1, 1), max_features=10_000)
    a_char = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", analyzer="char_wb",
                             ngram_range=(2, 6), max_features=50_000)
    b_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                             strip_accents="unicode", token_pattern=r"\w{1,}",
                             sublinear_tf=True, smooth_idf=True, use_idf=True,
                             max_features=200_000)
    a_word.fit(all_text)
    a_char.fit(all_text)
    b_word.fit(all_text)
    feats = {
        "a_word": (a_word.transform(train_text).tocsr(), a_word.transform(test_text).tocsr()),
        "a_char": (a_char.transform(train_text).tocsr(), a_char.transform(test_text).tocsr()),
        "b_word": (b_word.transform(train_text).tocsr(), b_word.transform(test_text).tocsr()),
    }
    return feats


def branch_a(x_tr, x_te, y):
    """word + char TF-IDF -> per-label logistic regression."""
    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = LogisticRegression(C=0.1, solver="sag", max_iter=1000, random_state=SEED)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [A] {lab} done", flush=True)
    return preds


def nb_ratio(x_tr, yj):
    """Naive-Bayes log-count ratio per feature column."""
    p = (1.0 + np.asarray(x_tr[yj == 1].sum(axis=0))) / (1.0 + (yj == 1).sum())
    q = (1.0 + np.asarray(x_tr[yj == 0].sum(axis=0))) / (1.0 + (yj == 0).sum())
    return np.log(p / q).ravel()


def branch_b(x_tr, x_te, y):
    """Naive-Bayes-weighted features (NB-LogReg) on word 1-2 gram TF-IDF."""
    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        yj = y[:, j]
        r = nb_ratio(x_tr, yj)
        clf = LogisticRegression(C=4.0, dual=True, solver="liblinear", random_state=SEED)
        clf.fit(x_tr.multiply(r).tocsr(), yj)
        preds[:, j] = clf.predict_proba(x_te.multiply(r).tocsr())[:, 1]
        print(f"  [B] {lab} done", flush=True)
    return preds


def branch_c(x_tr, x_te, y):
    """NB-LogReg on B's word 1-2-gram TF-IDF concatenated with A's char_wb TF-IDF.

    liblinear is capped at 100 outer iterations on purpose (bounds runtime on
    the 250k-column matrix); the resulting ConvergenceWarning is suppressed.
    """
    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        yj = y[:, j]
        r = nb_ratio(x_tr, yj)
        clf = LogisticRegression(C=4.0, solver="liblinear", dual=True, tol=1e-3,
                                 max_iter=100, random_state=SEED)
        xs_tr = x_tr.multiply(r).tocsr()
        xs_te = x_te.multiply(r).tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(xs_tr, yj)
        preds[:, j] = clf.predict_proba(xs_te)[:, 1]
        del xs_tr, xs_te
        print(f"  [C] {lab} done", flush=True)
    return preds


D_HIDDEN = 128
D_DROPOUT = 0.3
D_BATCH = 256
D_LR = 1e-3
D_MAX_EPOCHS = 4
D_VAL_FRAC = 0.1
D_MIN_VAL_AUC = 0.97
D_EPOCH_START_DEADLINE = 690.0  # seconds since t0: do not start a new epoch past this
D_START_DEADLINE = 720.0        # seconds since t0: skip D entirely past this


def _csr_to_bag(xb, torch):
    """CSR row block -> (indices, offsets, per_sample_weights) for EmbeddingBag."""
    indices = torch.from_numpy(xb.indices.astype(np.int64))
    offsets = torch.from_numpy(xb.indptr[:-1].astype(np.int64))
    weights = torch.from_numpy(np.ascontiguousarray(xb.data, dtype=np.float32))
    return indices, offsets, weights


def _predict_sparse(model, x, torch, bs=8192):
    out = np.zeros((x.shape[0], len(LABELS)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, x.shape[0], bs):
            xb = x[s:s + bs]
            logits = model(*_csr_to_bag(xb, torch))
            out[s:s + bs] = torch.sigmoid(logits).numpy()
    return out


def branch_d(x_tr, x_te, y, t0):
    """Multi-label sparse MLP: EmbeddingBag(sum) == sparse X @ W, shared ReLU hidden
    layer, 6 sigmoid outputs. Epoch chosen on a 10% held-out split. Returns None on
    any fallback (time guard, non-converged model)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(0)

    n_feat = x_tr.shape[1]
    fit_idx, val_idx = train_test_split(np.arange(x_tr.shape[0]), test_size=D_VAL_FRAC,
                                        random_state=0, stratify=y[:, 0])
    x_fit, x_val = x_tr[fit_idx], x_tr[val_idx]
    y_fit = y[fit_idx].astype(np.float32)
    y_val = y[val_idx]

    class SparseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.EmbeddingBag(num_embeddings=n_feat, embedding_dim=D_HIDDEN,
                                       mode="sum", sparse=True)
            nn.init.xavier_uniform_(self.emb.weight)
            self.b1 = nn.Parameter(torch.zeros(D_HIDDEN))
            self.drop = nn.Dropout(D_DROPOUT)
            self.out = nn.Linear(D_HIDDEN, len(LABELS))

        def forward(self, indices, offsets, weights):
            h = torch.relu(self.emb(indices, offsets, per_sample_weights=weights) + self.b1)
            return self.out(self.drop(h))

    model = SparseMLP()
    loss_fn = nn.BCEWithLogitsLoss()
    opt_sparse = torch.optim.SparseAdam([model.emb.weight], lr=D_LR)
    opt_dense = torch.optim.Adam([model.b1] + list(model.out.parameters()), lr=D_LR)
    rng = np.random.default_rng(0)

    best_auc, best_pred = -1.0, None
    n_fit = x_fit.shape[0]
    for epoch in range(1, D_MAX_EPOCHS + 1):
        if time.time() - t0 > D_EPOCH_START_DEADLINE:
            print(f"  [D] time guard: not starting epoch {epoch}", flush=True)
            break
        model.train()
        perm = rng.permutation(n_fit)
        running, nb = 0.0, 0
        for s in range(0, n_fit, D_BATCH):
            rows = perm[s:s + D_BATCH]
            xb = x_fit[rows]
            yb = torch.from_numpy(y_fit[rows])
            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            logits = model(*_csr_to_bag(xb, torch))
            loss = loss_fn(logits, yb)
            loss.backward()
            opt_sparse.step()
            opt_dense.step()
            running += loss.item()
            nb += 1
        for opt in (opt_sparse, opt_dense):
            for g in opt.param_groups:
                g["lr"] *= 0.5
        pv = _predict_sparse(model, x_val, torch)
        val_auc = float(np.mean([roc_auc_score(y_val[:, j], pv[:, j]) for j in range(len(LABELS))]))
        print(f"  [D] epoch {epoch} train_loss={running / max(nb, 1):.4f} "
              f"val_mean_auc={val_auc:.5f} at {time.time() - t0:.0f}s", flush=True)
        log_metric("d_train_loss", running / max(nb, 1), step=epoch)
        log_metric("d_val_mean_auc", val_auc, step=epoch)
        if val_auc > best_auc:
            best_auc = val_auc
            best_pred = _predict_sparse(model, x_te, torch)
        else:
            print(f"  [D] no improvement at epoch {epoch}, stopping", flush=True)
            break

    log_metric("d_best_val_mean_auc", best_auc, step=0)
    if best_pred is None or best_auc < D_MIN_VAL_AUC:
        print(f"  [D] discarded (best held-out mean AUC {best_auc:.4f})", flush=True)
        return None
    return best_pred.astype(np.float64)


def to_rank(p):
    """Column-wise rank scaled to (0, 1) — AUC only cares about ordering."""
    out = np.zeros_like(p)
    for j in range(p.shape[1]):
        out[:, j] = rankdata(p[:, j]) / (p.shape[0] + 1.0)
    return out


def main():
    t0 = time.time()
    log_params({
        "approach": "rank_avg_word_char_tfidf_logreg + nb_logreg + nb_logreg_word_char",
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
        "c_features": "b_word_200k + a_char_50k",
        "c_C": 4.0,
        "c_solver": "liblinear",
        "c_max_iter": 100,
        "d_features": "a_word+a_char+b_word",
        "d_hidden": D_HIDDEN,
        "d_dropout": D_DROPOUT,
        "d_batch": D_BATCH,
        "d_lr": "1e-3 halved each epoch",
        "d_max_epochs": D_MAX_EPOCHS,
        "d_val_frac": D_VAL_FRAC,
        "blend_weights": "1/4 each rank average of A, B, C, D",
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    feats = build_features(train_text, test_text)
    tf = time.time() - t0
    print(f"features built at {tf:.0f}s", flush=True)
    log_metric("features_seconds", tf, step=0)

    x_a_tr = hstack([feats["a_word"][0], feats["a_char"][0]]).tocsr()
    x_a_te = hstack([feats["a_word"][1], feats["a_char"][1]]).tocsr()
    pa = branch_a(x_a_tr, x_a_te, y)
    del x_a_tr, x_a_te
    ta = time.time() - t0
    print(f"branch A done at {ta:.0f}s", flush=True)
    log_metric("branch_a_seconds", ta - tf, step=0)

    pb = branch_b(feats["b_word"][0], feats["b_word"][1], y)
    tb = time.time() - t0
    print(f"branch B done at {tb:.0f}s", flush=True)
    log_metric("branch_b_seconds", tb - ta, step=0)

    x_c_tr = hstack([feats["b_word"][0], feats["a_char"][0]]).tocsr()
    x_c_te = hstack([feats["b_word"][1], feats["a_char"][1]]).tocsr()
    pc = branch_c(x_c_tr, x_c_te, y)
    del x_c_tr, x_c_te
    tc = time.time() - t0
    print(f"branch C done at {tc:.0f}s", flush=True)
    log_metric("branch_c_seconds", tc - tb, step=0)

    pd_ = None
    if time.time() - t0 > D_START_DEADLINE:
        print("branch D skipped: time guard", flush=True)
        del feats
    else:
        x_d_tr = hstack([feats["a_word"][0], feats["a_char"][0], feats["b_word"][0]]).tocsr().astype(np.float32)
        x_d_te = hstack([feats["a_word"][1], feats["a_char"][1], feats["b_word"][1]]).tocsr().astype(np.float32)
        del feats
        pd_ = branch_d(x_d_tr, x_d_te, y, t0)
        del x_d_tr, x_d_te
    td = time.time() - t0
    print(f"branch D done at {td:.0f}s ({'used' if pd_ is not None else 'discarded'})", flush=True)
    log_metric("branch_d_seconds", td - tc, step=0)
    log_metric("branch_d_used", 1.0 if pd_ is not None else 0.0, step=0)

    if pd_ is not None:
        blend = (to_rank(pa) + to_rank(pb) + to_rank(pc) + to_rank(pd_)) / 4.0
    else:
        blend = (to_rank(pa) + to_rank(pb) + to_rank(pc)) / 3.0

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
