"""Baseline for leaf-classification.

The 192 precomputed shape/margin/texture descriptors, standardised, fed to a plain
multinomial logistic regression. Targets the historical leaderboard median (~0.11
multi-class log loss); deliberately NOT state of the art: no feature selection, no
image features, no ensembling, no calibration, and the regularisation is left at a
single fixed value rather than tuned.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 1.0          # modest, untuned regularisation: lands the root near the leaderboard median


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all, y_all = featurize(train), train["species"].values
    x_test = featurize(test)

    scaler = StandardScaler().fit(x_all)
    x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all_s, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    print("local_holdout_logloss=%.5f" % log_loss(y_val, clf.predict_proba(x_val), labels=clf.classes_))

    clf.fit(x_all_s, y_all)
    proba = clf.predict_proba(x_test_s).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)
    out = pd.DataFrame(proba, columns=list(clf.classes_))
    for c in classes:
        if c not in out.columns:
            out[c] = 0.0
    sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
