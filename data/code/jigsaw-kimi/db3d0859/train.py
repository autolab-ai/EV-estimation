"""Single closed-form generative baseline for jigsaw-toxic-comment-classification-challenge.

One vectorizer + one model class, fit per label:
  CountVectorizer (word 1-2-grams, RAW counts, no TF-IDF,
                   min_df=3, max_df=0.9, max_features=200k)
  -> ComplementNB(alpha=1.0) per label, predict_proba for the test set.

The vectorizer is fit on train+test text, the same transductive convention
the previous three-branch code used. ComplementNB (Rennie et al., 2003)
estimates each class's parameters from the complement of that class, which
corrects the skewed weight estimates plain multinomial NB suffers under
class imbalance (threat is ~0.3% positives here). Closed-form: no iterative
solver, essentially no hyperparameters beyond smoothing, trains in seconds.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import contextlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB

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


def main():
    t0 = time.time()
    log_params({
        "approach": "count_word_1-2gram + ComplementNB per label",
        "seed": SEED,
        "vec": "CountVectorizer",
        "vec_analyzer": "word",
        "vec_ngram": "1-2",
        "vec_min_df": 3,
        "vec_max_df": 0.9,
        "vec_max_features": 200_000,
        "vec_strip_accents": "unicode",
        "vec_fit_on": "train+test",
        "model": "ComplementNB",
        "alpha": 1.0,
    })
    # ids are 16-char hex strings — force str so the merge below cannot mis-cast
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"), dtype={"id": str})
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"), dtype={"id": str})
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"), dtype={"id": str})

    train_text = train["comment_text"].fillna("unknown")
    test_text = test["comment_text"].fillna("unknown")
    y = train[LABELS].values

    all_text = pd.concat([train_text, test_text])
    vec = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.9,
                          max_features=200_000, strip_accents="unicode",
                          token_pattern=r"\w{1,}")
    vec.fit(all_text)
    x_tr = vec.transform(train_text)
    x_te = vec.transform(test_text)
    tv = time.time() - t0
    print(f"vectorized at {tv:.0f}s: train{tuple(x_tr.shape)} test{tuple(x_te.shape)}", flush=True)
    log_metric("vectorize_seconds", tv, step=0)

    preds = np.zeros((x_te.shape[0], len(LABELS)))
    for j, lab in enumerate(LABELS):
        clf = ComplementNB(alpha=1.0)
        clf.fit(x_tr, y[:, j])
        preds[:, j] = clf.predict_proba(x_te)[:, 1]
        print(f"  [CNB] {lab} done at {time.time() - t0:.0f}s", flush=True)
    tf = time.time() - t0
    log_metric("cnb_fit_seconds", tf - tv, step=0)

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
