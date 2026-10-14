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
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

_MLFLOW_OK = mlflow is not None


def _log_params(d):
    global _MLFLOW_OK
    if not _MLFLOW_OK:
        return
    try:
        mlflow.log_params(d)
    except Exception as e:  # tracking is best-effort
        print(f"mlflow disabled: {e}")
        _MLFLOW_OK = False


def _log_metric(k, v, step=None):
    global _MLFLOW_OK
    if not _MLFLOW_OK:
        return
    try:
        mlflow.log_metric(k, float(v), step=step)
    except Exception as e:
        print(f"mlflow disabled: {e}")
        _MLFLOW_OK = False

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0


def featurize(train_text, test_text):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, max_features=60_000)
    x_train = hstack([word.fit_transform(train_text),
                      char.fit_transform(train_text)]).tocsr()
    x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
    return x_train, x_test


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))

    x_all, x_test = featurize(train["text"], test["text"])
    y_all = np.asarray(train["author"].tolist(), dtype=object)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(x_all, y_all))
    n_tr, n_te = x_all.shape[0], x_test.shape[0]

    # ---- LEVEL 1, base model 1: LR on word+char TF-IDF (fold-parallel C selection) ----
    Cs = [1, 3, 10, 30, 100]
    _log_params({"Cs": str(Cs), "cv_folds": 5, "solver": "lbfgs", "max_iter": 3000,
                 "nb_alphas": str([0.01, 0.03, 0.1, 0.3, 1.0]),
                 "meta_Cs": str([0.03, 0.1, 0.3, 1, 3, 10])})

    def fold_scores(tr_idx, va_idx):
        out = []
        for c in Cs:
            m = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=SEED)
            m.fit(x_all[tr_idx], y_all[tr_idx])
            assert list(m.classes_) == CLASSES
            p_va = m.predict_proba(x_all[va_idx])
            p_te = m.predict_proba(x_test)
            out.append((log_loss(y_all[va_idx], p_va, labels=m.classes_), p_va, p_te))
        return out

    fold_out = Parallel(n_jobs=5)(delayed(fold_scores)(tr, va) for tr, va in folds)
    scores = np.array([[r[0] for r in fo] for fo in fold_out])
    mean_ll = scores.mean(axis=0)
    for c, ll in zip(Cs, mean_ll):
        print(f"cv_logloss C={c:g}: {ll:.5f}")
        _log_metric("cv_logloss", ll, step=int(c))
    best_i = int(mean_ll.argmin())
    best_c = float(Cs[best_i])
    oof_lr = np.zeros((n_tr, 3))
    test_lr = np.zeros((n_te, 3))
    for (tr_idx, va_idx), fo in zip(folds, fold_out):
        oof_lr[va_idx] = fo[best_i][1]
        test_lr += fo[best_i][2] / len(folds)
    lr_oof_ll = log_loss(y_all, oof_lr, labels=CLASSES)
    print(f"chosen_C={best_c:g}  lr_oof_logloss={lr_oof_ll:.5f}")
    _log_metric("chosen_C", best_c)
    _log_metric("lr_oof_logloss", lr_oof_ll)

    # ---- LEVEL 1, base models 2 & 3: MultinomialNB on raw word / char counts ----
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0]

    def nb_stack(name, xc_all, xc_test):
        best = None
        for a in alphas:
            oof = np.zeros((n_tr, 3))
            te = np.zeros((n_te, 3))
            for tr_idx, va_idx in folds:
                m = MultinomialNB(alpha=a)
                m.fit(xc_all[tr_idx], y_all[tr_idx])
                assert list(m.classes_) == CLASSES
                oof[va_idx] = m.predict_proba(xc_all[va_idx])
                te += m.predict_proba(xc_test) / len(folds)
            ll = log_loss(y_all, oof, labels=CLASSES)
            print(f"{name} alpha={a:g}: oof_logloss={ll:.5f}")
            if best is None or ll < best[0]:
                best = (ll, a, oof, te)
        print(f"{name} chosen_alpha={best[1]:g}  oof_logloss={best[0]:.5f}")
        _log_metric(f"{name}_alpha", best[1])
        _log_metric(f"{name}_oof_logloss", best[0])
        return best[2], best[3]

    cv_word = CountVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
    xw_all = cv_word.fit_transform(train["text"])
    xw_test = cv_word.transform(test["text"])
    oof_nbw, test_nbw = nb_stack("nb_word", xw_all, xw_test)

    cv_char = CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                              max_features=200_000, lowercase=True)
    xc_all = cv_char.fit_transform(train["text"])
    xc_test = cv_char.transform(test["text"])
    oof_nbc, test_nbc = nb_stack("nb_char", xc_all, xc_test)

    # ---- LEVEL 2: log-linear meta-learner on log-probabilities (learned geometric blend) ----
    def meta_feats(*ps):
        return np.hstack([np.log(np.clip(p, 1e-6, 1.0)) for p in ps])

    z_all = meta_feats(oof_lr, oof_nbw, oof_nbc)
    z_test = meta_feats(test_lr, test_nbw, test_nbc)
    meta_Cs = [0.03, 0.1, 0.3, 1, 3, 10]
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    meta_ll = []
    for c in meta_Cs:
        lls = []
        for tr_idx, va_idx in skf2.split(z_all, y_all):
            m = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=SEED)
            m.fit(z_all[tr_idx], y_all[tr_idx])
            lls.append(log_loss(y_all[va_idx], m.predict_proba(z_all[va_idx]), labels=m.classes_))
        meta_ll.append(float(np.mean(lls)))
        print(f"meta_cv_logloss C={c:g}: {meta_ll[-1]:.5f}")
        _log_metric("meta_cv_logloss", meta_ll[-1], step=int(round(c * 100)))
    mi = int(np.argmin(meta_ll))
    meta_c = float(meta_Cs[mi])
    cv_ll = meta_ll[mi]
    print(f"meta_chosen_C={meta_c:g}")
    print(f"local_holdout_logloss={cv_ll:.5f}")
    _log_metric("meta_chosen_C", meta_c)
    _log_metric("local_holdout_logloss", cv_ll)

    clf = LogisticRegression(C=meta_c, solver="lbfgs", max_iter=1000, random_state=SEED)
    clf.fit(z_all, y_all)
    proba = clf.predict_proba(z_test).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)

    sub = pd.DataFrame(proba, columns=list(clf.classes_))
    sub.insert(0, "id", test["id"])
    sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub[["id"] + CLASSES].to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    if mlflow is not None:
        try:
            with mlflow.start_run():
                main()
        except Exception as e:
            if os.path.exists(os.path.join("submission", "submission.csv")):
                print(f"mlflow error after run: {e}")
            else:
                print(f"mlflow start failed ({e}); running untracked")
                _MLFLOW_OK = False
                main()
    else:
        main()
