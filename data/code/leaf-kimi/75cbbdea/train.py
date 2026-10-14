"""Cross-validated-regularisation variant of the leaf-classification baseline.

Identical pipeline to the root baseline (192 precomputed shape/margin/texture
descriptors, standardised, multinomial logistic regression, out-of-fold-selected
uniform shrinkage), except the logistic-regression regularisation strength C is no
longer fixed at 1.0: it is selected by 5-fold out-of-fold log loss over
np.logspace(-2, 3, 11) and the winner is used both in the 10-fold shrinkage
selection and in the final full-data fit.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 1.0          # fixed regularisation, kept only for the local-holdout diagnostic (baseline-comparable)
C_GRID = np.logspace(-2, 3, 11)   # model-selection grid for the LR regularisation strength


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all, y_all = featurize(train), train["species"].to_numpy()
    x_test = featurize(test)

    # Model selection: pick the LR regularisation strength C by 5-fold
    # out-of-fold log loss on the raw features (the pipeline re-scales per fold).
    cv_ms = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_losses_c = []
    for cand in C_GRID:
        pipe_ms = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=float(cand), max_iter=5000, random_state=SEED)),
        ])
        oof_c = cross_val_predict(pipe_ms, x_all, y_all, cv=cv_ms, method="predict_proba", n_jobs=-1)
        oof_losses_c.append(log_loss(y_all, oof_c))
    best_c_idx = int(np.argmin(oof_losses_c))
    C_best = float(C_GRID[best_c_idx])
    print("c_best=%.6g oof_logloss_c=%.5f" % (C_best, oof_losses_c[best_c_idx]))

    scaler = StandardScaler().fit(x_all)
    x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all_s, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
    clf.fit(x_tr, y_tr)
    print("local_holdout_logloss=%.5f" % log_loss(y_val, clf.predict_proba(x_val), labels=clf.classes_))

    # Select shrinkage strength alpha (toward the uniform prior over species) by
    # out-of-fold validation on the raw features; the pipeline re-scales per fold.
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C_best, max_iter=5000, random_state=SEED)),
    ])
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, x_all, y_all, cv=cv, method="predict_proba")
    n_classes = len(classes)
    alphas = np.linspace(0.0, 0.5, 51)
    oof_losses = [log_loss(y_all, (1.0 - a) * oof + a / n_classes, labels=classes) for a in alphas]
    best_idx = int(np.argmin(oof_losses))
    alpha = float(alphas[best_idx])
    print("oof_alpha=%.4f oof_logloss=%.5f" % (alpha, oof_losses[best_idx]))

    clf_final = LogisticRegression(C=C_best, max_iter=5000, random_state=SEED)
    clf_final.fit(x_all_s, y_all)
    proba = clf_final.predict_proba(x_test_s).astype(np.float64)
    proba = (1.0 - alpha) * proba + alpha / n_classes
    proba = np.clip(proba, 1e-15, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    out = pd.DataFrame(proba, columns=list(clf_final.classes_))
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
