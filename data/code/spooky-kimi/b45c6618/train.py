"""Complement Naive Bayes on the baseline word+char TF-IDF stack for
spooky-author-identification.

featurize() is exactly the baseline's; the hybrid head is replaced by a
single multiclass ComplementNB(alpha=1.0) fit directly on the non-negative
sparse TF-IDF matrix and string labels. ComplementNB estimates each class
from the complement of the other classes, a standard text-classification
variant for uneven class counts, and emits multiclass probabilities from one
fitted model. Writes submission.csv and submission/submission.csv; the hill's
evaluator grades the latter.
"""

import contextlib
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

try:
    import mlflow
    _mlflow_ok = True
except Exception as _exc:  # tracking is best-effort, never fatal
    print(f"mlflow unavailable, tracking disabled: {_exc}")
    _mlflow_ok = False


def _start_run():
    if not _mlflow_ok:
        return contextlib.nullcontext()
    try:
        return mlflow.start_run()
    except Exception as exc:
        print(f"mlflow.start_run failed, running untracked: {exc}")
        return contextlib.nullcontext()


def _log_params(params):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        mlflow.log_params(params)
    except Exception as exc:
        print(f"mlflow log_params failed, disabling tracking: {exc}")
        _mlflow_ok = False


def _log_metric(name, value, **kwargs):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        mlflow.log_metric(name, value, **kwargs)
    except Exception as exc:
        print(f"mlflow log_metric failed, disabling tracking: {exc}")
        _mlflow_ok = False


COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0
NB_ALPHA = 1.0  # ComplementNB smoothing


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def nbsvm_proba(x_fit, y_fit, x_score):
    """Fit one multiclass ComplementNB on (x_fit, y_fit) and score x_score.

    Returns predict_proba columns permuted into CLASSES order via clf.classes_.
    """
    clf = ComplementNB(alpha=NB_ALPHA)
    clf.fit(x_fit, y_fit)
    proba = clf.predict_proba(x_score).astype(np.float64)
    order = [list(clf.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = np.asarray(train["author"], dtype=str)  # plain ndarray: pandas 3 gives a pyarrow-backed array sklearn cannot index

    with _start_run():
        _log_params({
            "model": "complement-nb",
            "nb_alpha": NB_ALPHA,
            "word_ngram_range": "1-2",
            "char_ngram_range": "2-5",
            "char_max_features": 60000,
            "seed": SEED,
        })

        # quick local estimate on a 10% holdout
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )
        val_proba = nbsvm_proba(x_tr, y_tr, x_val)
        holdout = log_loss(y_val, val_proba, labels=CLASSES)
        print(f"local_holdout_logloss={holdout:.5f}")
        _log_metric("local_holdout_logloss", holdout)

        # refit on everything for the real submission
        proba = nbsvm_proba(x_all, y_all, x_test)

        sub = pd.DataFrame(proba, columns=CLASSES)
        sub.insert(0, "id", test["id"])
        sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
        os.makedirs('submission', exist_ok=True)  # the hill grades this directory
        sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
