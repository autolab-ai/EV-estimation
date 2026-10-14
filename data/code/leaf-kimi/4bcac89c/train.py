"""Leaf-classification: 192 precomputed shape/margin/texture descriptors,
standardised, fed to multinomial logistic regression whose L2 regularisation
strength C is selected by 10-fold stratified cross-validation (neg log loss)
over a log-spaced grid.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0


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

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 25),
        penalty="l2",
        solver="lbfgs",
        max_iter=10000,
        scoring="neg_log_loss",
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(x_all_s, y_all)
    print("selected_C=%.6g" % clf.C_[0])
    print("mean_cv_neg_log_loss=%.5f" % clf.scores_[clf.classes_[0]].mean(axis=0)[list(clf.Cs_).index(clf.C_[0])])
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
