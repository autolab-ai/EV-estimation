"""Spooky-author-identification: 5-fold out-of-fold stacking ensemble.

Base models (each yielding 3-class probabilities via 5-fold OOF):
  M1: LogisticRegression on word TF-IDF (1-2 grams)
  M2: LogisticRegression on char TF-IDF (1-5 grams)
  M3: MultinomialNB on word counts (1-2 grams), alpha tuned on OOF log loss
  M4: MultinomialNB on char counts (1-4 grams), alpha tuned on OOF log loss
Meta-learner: LogisticRegression on the log-probabilities of the four base models,
C chosen by 5-fold CV log loss. Test predictions of each base model are the mean
of the 5 fold models' predictions (no refit on all training data).
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore", category=ConvergenceWarning)

COMP = "spooky-author-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 0
N_FOLDS = 5
NB_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]
META_CS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# ---------------------------------------------------------------- mlflow (best effort)
class _Tracker:
    def __init__(self):
        self.ok = False
        self.mlflow = None
        try:
            import mlflow  # noqa

            self.mlflow = mlflow
            self.ok = True
        except Exception as e:  # pragma: no cover
            print(f"[mlflow] unavailable: {e}")

    def start(self):
        if not self.ok:
            return _Noop()
        try:
            return self.mlflow.start_run()
        except Exception as e:
            print(f"[mlflow] start_run failed, continuing untracked: {e}")
            self.ok = False
            return _Noop()

    def log_params(self, d):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:
            print(f"[mlflow] log_params failed, disabling tracking: {e}")
            self.ok = False

    def log_metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print(f"[mlflow] log_metric failed, disabling tracking: {e}")
            self.ok = False


class _Noop:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------- OOF machinery
def _fit_fold(make_model, X, y, X_test, tr_idx, val_idx):
    m = make_model()
    m.fit(X[tr_idx], y[tr_idx])
    assert list(m.classes_) == CLASSES, m.classes_
    return val_idx, m.predict_proba(X[val_idx]), m.predict_proba(X_test)


def oof_predict(make_model, X, y, X_test, skf, n_jobs=1):
    splits = list(skf.split(X, y))
    if n_jobs == 1:
        results = [_fit_fold(make_model, X, y, X_test, tr, va) for tr, va in splits]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_fold)(make_model, X, y, X_test, tr, va) for tr, va in splits
        )
    oof = np.zeros((X.shape[0], len(CLASSES)))
    test = np.zeros((X_test.shape[0], len(CLASSES)))
    for val_idx, pv, pt in results:
        oof[val_idx] = pv
        test += pt / len(results)
    return oof, test


def main():
    tracker = _Tracker()
    with tracker.start():
        tracker.log_params({
            "model": "stack5fold_lrw_lrc_nbw_nbc",
            "n_folds": N_FOLDS,
            "base_lr_C": 10.0,
            "nb_alphas": str(NB_ALPHAS),
            "meta_Cs": str(META_CS),
            "seed": SEED,
        })

        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
        y = train["author"].values
        tr_text, te_text = train["text"], test["text"]

        # 1) feature matrices (unsupervised vectorizers fit once on full train text)
        word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        char_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                                     min_df=2, sublinear_tf=True)
        word_cnt = CountVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
        char_cnt = CountVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False, min_df=2)

        Xw, Xw_te = word_tfidf.fit_transform(tr_text).tocsr(), word_tfidf.transform(te_text).tocsr()
        Xc, Xc_te = char_tfidf.fit_transform(tr_text).tocsr(), char_tfidf.transform(te_text).tocsr()
        Cw, Cw_te = word_cnt.fit_transform(tr_text).tocsr(), word_cnt.transform(te_text).tocsr()
        Cc, Cc_te = char_cnt.fit_transform(tr_text).tocsr(), char_cnt.transform(te_text).tocsr()
        print(f"features: Xw={Xw.shape} Xc={Xc.shape} Cw={Cw.shape} Cc={Cc.shape}", flush=True)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        def make_lr():
            return LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000, random_state=SEED)

        # 2/3) base models via OOF
        oofs, tests = [], []
        for name, X, X_te in [("lr_word", Xw, Xw_te), ("lr_char", Xc, Xc_te)]:
            oof, te = oof_predict(make_lr, X, y, X_te, skf, n_jobs=N_FOLDS)
            ll = log_loss(y, oof, labels=CLASSES)
            print(f"oof_logloss[{name}]={ll:.5f}", flush=True)
            tracker.log_metric(f"oof_logloss_{name}", ll)
            oofs.append(oof)
            tests.append(te)

        for name, X, X_te in [("nb_word", Cw, Cw_te), ("nb_char", Cc, Cc_te)]:
            best = None
            for a in NB_ALPHAS:
                oof, te = oof_predict(lambda a=a: MultinomialNB(alpha=a), X, y, X_te, skf)
                ll = log_loss(y, oof, labels=CLASSES)
                print(f"  {name} alpha={a}: oof_logloss={ll:.5f}", flush=True)
                if best is None or ll < best[0]:
                    best = (ll, a, oof, te)
            ll, a, oof, te = best
            print(f"oof_logloss[{name}]={ll:.5f} (alpha={a})", flush=True)
            tracker.log_metric(f"oof_logloss_{name}", ll)
            tracker.log_params({f"{name}_alpha": a})
            oofs.append(oof)
            tests.append(te)

        # 4) meta features: log-probabilities of every base model
        def logp(p):
            return np.log(np.clip(p, 1e-5, 1.0))

        Z_tr = np.hstack([logp(p) for p in oofs])
        Z_te = np.hstack([logp(p) for p in tests])

        # 5) meta-learner with C chosen by CV log loss
        meta_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        best_c, best_ll = None, np.inf
        for i, C in enumerate(META_CS):
            meta = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=SEED)
            p = cross_val_predict(meta, Z_tr, y, cv=meta_cv, method="predict_proba")
            ll = log_loss(y, p, labels=CLASSES)
            print(f"  meta C={C}: cv_logloss={ll:.5f}", flush=True)
            tracker.log_metric("meta_cv_logloss", ll, step=i)
            if ll < best_ll:
                best_c, best_ll = C, ll
        print(f"meta_best_C={best_c} local_cv_logloss={best_ll:.5f}", flush=True)
        tracker.log_params({"meta_best_C": best_c})
        tracker.log_metric("local_holdout_logloss", best_ll)

        meta = LogisticRegression(C=best_c, solver="lbfgs", max_iter=2000, random_state=SEED)
        meta.fit(Z_tr, y)
        assert list(meta.classes_) == CLASSES
        proba = meta.predict_proba(Z_te).astype(np.float64)
        proba /= proba.sum(axis=1, keepdims=True)  # grader requires row sums == 1 (1e-6 tol)

        sub = pd.DataFrame(proba, columns=CLASSES)
        sub.insert(0, "id", test["id"])
        sub[["id"] + CLASSES].to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub[["id"] + CLASSES].to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
