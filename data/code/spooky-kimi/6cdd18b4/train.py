"""fastText supervised classifier for spooky-author-identification.

Replaces the TF-IDF + LogisticRegression baseline pipeline with Facebook's
fastText supervised text classifier (pure CPU, trains in well under two
minutes on this data). Text is lowercased, punctuation is padded with
spaces, and whitespace is collapsed before training/prediction.

Protocol: same stratified 10% holdout (random_state=0) for a local log-loss
estimate, then a full-data retrain for the submission. Writes
submission.csv and submission/submission.csv; the hill's evaluator grades
the latter.
"""

import contextlib
import os
import re
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0

FT_PARAMS = dict(
    lr=0.5, dim=100, epoch=25, wordNgrams=2, minn=2, maxn=5,
    loss="softmax", minCount=2, thread=4,
)

_PUNCT_RE = re.compile(r"([.,!?;:'\"()\-])")
_WS_RE = re.compile(r"\s+")


def preprocess(text):
    text = str(text).lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return _WS_RE.sub(" ", text).strip()


def balance_classes(x, y):
    """Random-oversample minority classes up to the majority class count M.

    Draws (M - count) extra row indices with replacement per under-sized
    class, concatenates them onto the originals, and shuffles everything
    with the same RandomState(0).
    """
    rng = np.random.RandomState(SEED)
    classes, counts = np.unique(y, return_counts=True)
    m = counts.max()
    extra_idx = []
    for cls, cnt in zip(classes, counts):
        if cnt < m:
            cls_idx = np.flatnonzero(y == cls)
            extra_idx.append(rng.choice(cls_idx, size=m - cnt, replace=True))
    all_idx = np.concatenate([np.arange(len(y))] + extra_idx)
    all_idx = rng.permutation(all_idx)
    return x[all_idx], y[all_idx]


def write_fasttext_file(path, texts, labels):
    with open(path, "w", encoding="utf-8") as f:
        for t, y in zip(texts, labels):
            f.write(f"__label__{y} {t}\n")


def predict_proba(model, texts):
    """k=3 predictions mapped onto the [EAP, HPL, MWS] column order."""
    labels, probs = model.predict(list(texts), k=3)
    out = np.full((len(texts), len(CLASSES)), 1e-6, dtype=np.float64)
    for i, (labs, ps) in enumerate(zip(labels, probs)):
        for lab, p in zip(labs, ps):
            out[i, CLASSES.index(lab.replace("__label__", ""))] = p
    out /= out.sum(axis=1, keepdims=True)
    return out


# --- best-effort MLflow telemetry: never allowed to break the run ---------
class _MLflow:
    def __init__(self):
        self.enabled = False
        self._mlflow = None
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return  # no preconfigured tracking server; stay a no-op
        try:
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
            import mlflow

            self._mlflow = mlflow
            self.enabled = True
        except Exception as e:  # pragma: no cover
            print(f"mlflow import warning: {e}")

    def start_run(self):
        if not self.enabled:
            return contextlib.nullcontext()
        try:
            return self._mlflow.start_run()
        except Exception as e:
            print(f"mlflow start_run warning: {e}")
            self.enabled = False
            return contextlib.nullcontext()

    def _safe(self, method, *args, **kwargs):
        if not self.enabled:
            return
        try:
            getattr(self._mlflow, method)(*args, **kwargs)
        except Exception as e:
            print(f"mlflow {method} warning: {e}")
            self.enabled = False

    def log_params(self, params):
        self._safe("log_params", params)

    def log_metric(self, key, value, step=None):
        self._safe("log_metric", key, value, step=step)


def main():
    import fasttext

    mlf = _MLflow()
    with mlf.start_run():
        mlf.log_params({"model": "fasttext_supervised", "balance_oversample": True, **FT_PARAMS})

        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

        x_all = train["text"].map(preprocess).to_numpy()
        y_all = train["author"].to_numpy()

        # quick local estimate on the same stratified 10% holdout as before
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )
        # balance the training portion only; x_val/y_val stay natural so the
        # local log-loss estimate reflects the real class distribution
        x_tr_bal, y_tr_bal = balance_classes(x_tr, y_tr)
        with tempfile.TemporaryDirectory() as tmp:
            holdout_file = os.path.join(tmp, "train_holdout.txt")
            write_fasttext_file(holdout_file, x_tr_bal, y_tr_bal)
            model = fasttext.train_supervised(
                input=holdout_file, verbose=0, **FT_PARAMS
            )
            val_proba = predict_proba(model, x_val)

            full_file = os.path.join(tmp, "train_full.txt")
            x_all_bal, y_all_bal = balance_classes(x_all, y_all)
            write_fasttext_file(full_file, x_all_bal, y_all_bal)
            full_model = fasttext.train_supervised(
                input=full_file, verbose=0, **FT_PARAMS
            )

        holdout_loss = log_loss(y_val, val_proba, labels=CLASSES)
        print(f"local_holdout_logloss={holdout_loss:.5f}")
        mlf.log_metric("local_holdout_logloss", holdout_loss)

        # full-data model produces the real submission
        proba = predict_proba(full_model, test["text"].map(preprocess).to_numpy())
        sub = pd.DataFrame(proba, columns=CLASSES)
        sub.insert(0, "id", test["id"])
        sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub[["id"] + CLASSES].to_csv(
            os.path.join("submission", "submission.csv"), index=False
        )
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
