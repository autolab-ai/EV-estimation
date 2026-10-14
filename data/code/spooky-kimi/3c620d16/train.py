"""Baseline for spooky-author-identification.

A solid mid-competition solution: word + character TF-IDF features stacked,
logistic regression with moderate regularization. Targets the historical
leaderboard median (~0.42 log loss); deliberately NOT state of the art.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0


def style_features(texts):
    """14 per-sentence stylometric features, as a dense float64 array.

    Word-length stats are 0 for sentences with no whitespace-separated tokens.
    """
    s = pd.Series(np.asarray(list(texts)), dtype="object").fillna("").astype(str)

    char_count = s.str.len().astype(np.float64)
    tokens = s.str.split()
    word_count = tokens.str.len().astype(np.float64)

    # word-length stats via one long exploded Series (vectorized)
    wl = tokens.explode().str.len().astype(np.float64)
    by_row = wl.groupby(level=0)
    idx = pd.RangeIndex(len(s))
    mean_word_len = by_row.mean().reindex(idx).fillna(0.0)
    std_word_len = by_row.std().reindex(idx).fillna(0.0)  # NaN for 0/1 tokens -> 0
    max_word_len = by_row.max().reindex(idx).fillna(0.0)

    upper_frac = np.where(char_count > 0,
                          s.str.count(r"[A-Z]").astype(np.float64) / np.maximum(char_count, 1.0),
                          0.0)
    digit_count = s.str.count(r"\d").astype(np.float64)
    commas = s.str.count(",").astype(np.float64)
    semicolons = s.str.count(";").astype(np.float64)
    colons = s.str.count(":").astype(np.float64)
    questions = s.str.count(r"\?").astype(np.float64)
    exclaims = s.str.count("!").astype(np.float64)
    hyphens = s.str.count("-").astype(np.float64)
    dquotes = s.str.count('"').astype(np.float64)

    return np.column_stack([
        char_count, word_count, mean_word_len.to_numpy(), std_word_len.to_numpy(),
        max_word_len.to_numpy(), upper_frac, digit_count, commas, semicolons,
        colons, questions, exclaims, hyphens, dquotes,
    ]).astype(np.float64)


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    st_train = style_features(train_text)
    style_scaler = MinMaxScaler().fit(st_train)  # NB needs non-negative inputs
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text),
                      csr_matrix(style_scaler.transform(st_train))]).tocsr()
    x_test = hstack([word.transform(test_text),
                     char.transform(test_text),
                     csr_matrix(style_scaler.transform(style_features(test_text)))]).tocsr()
    return x_train, x_test


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = train["author"].values

    # quick local estimate on a 10% holdout
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )
    clf = MultinomialNB(alpha=0.1)
    clf.fit(x_tr, y_tr)
    print(f"local_holdout_logloss={log_loss(y_val, clf.predict_proba(x_val)):.5f}")

    # refit on everything for the real submission
    clf.fit(x_all, y_all)
    proba = clf.predict_proba(x_test).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)

    sub = pd.DataFrame(proba, columns=list(clf.classes_))
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
