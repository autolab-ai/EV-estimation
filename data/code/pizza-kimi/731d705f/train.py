"""Baseline for random-acts-of-pizza.

Word TF-IDF over the request text plus the nine requester-history numbers that the test
split also carries, fed to a plain logistic regression. This branch adds one change: a
fixed block of ~20 hand-crafted per-document statistics (length/structure, style and
punctuation, pronoun/gratitude flags, VADER sentiment, Flesch readability) computed from
the raw title and edit-aware body, scaled together with the 9 numeric columns and hstacked
onto the TF-IDF matrix.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

COMP = "random-acts-of-pizza"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 0.3                      # modest, untuned regularisation
MAX_FEATURES = 5_000         # unigrams only
NUM_COLS = ["requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request"]

_ANALYZER = SentimentIntensityAnalyzer()   # instantiated once, lexicon is bundled
_SENT_SPLIT = re.compile(r"[.!?]+")
_MONEY_RE = re.compile(r"\bmoney|cash|dollar|rent|paycheck|bill(s)?\b", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|we|us|our)\b", re.IGNORECASE)
_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
_WORD_RES = {w: re.compile(r"\b" + w + r"\b", re.IGNORECASE)
             for w in ("please", "thank", "appreciate", "grateful", "sorry")}
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def load(name):
    with open(os.path.join(PUBLIC, name)) as f:
        return pd.DataFrame(json.load(f))


def text_of(df):
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("")
    body = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * len(df)))).fillna("")
    return (title + " " + body).astype(str)


def _syllables(word):
    return max(1, len(_VOWEL_GROUP_RE.findall(word)))


def _row_features(title, body):
    words = body.split()
    n_words = len(words)
    sentences = max(1, len([s for s in _SENT_SPLIT.split(body) if s.strip()]))
    n_chars = len(body)
    avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    title_words = title.split()

    n_bang = body.count("!")
    n_quest = body.count("?")
    caps_frac = (sum(1 for w in words if len(w) > 1 and w.isupper()) / n_words) if n_words else 0.0
    money_flag = float(("$" in body) or bool(_MONEY_RE.search(body)))
    low = body.lower()
    link_flag = float(("http" in low) or ("www" in low))
    digit_flag = float(any(ch.isdigit() for ch in body))

    first_pp = len(_FIRST_PERSON_RE.findall(body)) / max(n_words, 1)
    second_pp = len(_SECOND_PERSON_RE.findall(body)) / max(n_words, 1)
    word_flags = [float(bool(rx.search(body))) for rx in _WORD_RES.values()]

    sent_body = _ANALYZER.polarity_scores(body)
    sent_title = _ANALYZER.polarity_scores(title)

    n_syll = sum(_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (n_syll / max(n_words, 1))

    return [np.log1p(n_words), np.log1p(n_chars), np.log1p(sentences),
            avg_word_len, np.log1p(len(title)), np.log1p(len(title_words)),
            float(n_bang), float(n_quest), caps_frac, money_flag, link_flag, digit_flag,
            first_pp, second_pp, *word_flags,
            sent_body["compound"], sent_body["pos"], sent_body["neg"],
            sent_title["compound"],
            flesch]


def doc_stats(df):
    """Per-document engineered statistics, shared by train and test."""
    title = df.get("request_title", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    if "request_text_edit_aware" in df.columns:
        body = df["request_text_edit_aware"].fillna("").astype(str)
    else:
        body = df.get("request_text", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    return np.asarray([_row_features(t, b) for t, b in zip(title, body)], dtype=np.float64)


def featurize(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                          sublinear_tf=True, stop_words="english",
                          strip_accents="unicode")
    xt = vec.fit_transform(text_of(train))
    xs = vec.transform(text_of(test))
    scaler = StandardScaler()
    nt = scaler.fit_transform(np.hstack([np.log1p(train[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)),
                                         doc_stats(train)]))
    ns = scaler.transform(np.hstack([np.log1p(test[NUM_COLS].to_numpy(dtype=np.float64).clip(min=0)),
                                     doc_stats(test)]))
    return hstack([xt, csr_matrix(nt)]).tocsr(), hstack([xs, csr_matrix(ns)]).tocsr()


def main():
    train, test = load("train.json"), load("test.json")
    y = train["requester_received_pizza"].astype(int).to_numpy()
    x_all, x_test = featurize(train, test)

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y, test_size=0.2, stratify=y, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    print("local_holdout_auc=%.5f" % roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1]))

    clf.fit(x_all, y)
    p = clf.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame({"request_id": test["request_id"].values, "requester_received_pizza": p})
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
