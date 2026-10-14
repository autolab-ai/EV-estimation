"""Leaf-classification: 192 precomputed shape/margin/texture descriptors,
standardised, classified by a fixed 3-model probability-averaging ensemble
(multinomial logistic regression, distance-weighted kNN, shrinkage LDA) whose
blend weights are learned from out-of-fold (OOF) predictions on the training
data only.

Hyperparameters are picked once on the full standardised training set
(logistic C via LogisticRegressionCV, k via GridSearchCV over 1..40, LDA
untuned); OOF probabilities from a fresh 10-fold split drive an SLSQP
optimisation of the convex blend weights against multi-class log loss.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---- best-effort MLflow logging: a tracking failure must never kill the run ----
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class _NoopCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class Tracker:
    def __init__(self):
        self.ok = mlflow is not None

    def start(self):
        if not self.ok:
            return _NoopCtx()
        try:
            return mlflow.start_run()
        except Exception as e:
            print("mlflow disabled: %s" % e)
            self.ok = False
            return _NoopCtx()

    def _guard(self, fn, *args, **kwargs):
        if not self.ok:
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print("mlflow disabled: %s" % e)
            self.ok = False

    def log_params(self, d):
        self._guard(mlflow.log_params, d)

    def log_metric(self, key, value, step=None):
        self._guard(mlflow.log_metric, key, float(value), step=step)


COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
MODEL_NAMES = ("lr", "knn", "lda")


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def make_models(selected_C, best_k):
    return {
        "lr": LogisticRegression(C=selected_C, max_iter=10000),
        "knn": KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="euclidean"),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }


def main():
    tracker = Tracker()
    with tracker.start():
        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
        sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
        classes = [c for c in sample.columns if c != "id"]

        x_all, y_all = featurize(train), np.asarray(train["species"], dtype=str)
        x_test = featurize(test)

        scaler = StandardScaler().fit(x_all)
        x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

        # ---- hyperparameters, picked once on the full standardised training set ----
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        lr_cv = LogisticRegressionCV(
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
        lr_cv.fit(x_all_s, y_all)
        selected_C = float(lr_cv.C_[0])
        print("selected_C=%.6g" % selected_C)

        knn_cv = GridSearchCV(
            KNeighborsClassifier(weights="distance", metric="euclidean"),
            param_grid={"n_neighbors": list(range(1, 41))},
            scoring="neg_log_loss",
            cv=cv,
            refit=True,
            n_jobs=-1,
        )
        knn_cv.fit(x_all_s, y_all)
        best_k = int(knn_cv.best_params_["n_neighbors"])
        print("best_k=%d" % best_k)

        # ---- OOF probability matrices (one per model) on a fresh 10-fold split ----
        cls_list = list(np.unique(y_all))
        cls_idx = {c: i for i, c in enumerate(cls_list)}
        n, m = x_all.shape[0], len(cls_list)
        oof = {name: np.zeros((n, m)) for name in MODEL_NAMES}
        oof_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        for tr_idx, va_idx in oof_cv.split(x_all, y_all):
            fold_scaler = StandardScaler().fit(x_all[tr_idx])  # refit per fold: no leakage
            x_tr, x_va = fold_scaler.transform(x_all[tr_idx]), fold_scaler.transform(x_all[va_idx])
            for name, model in make_models(selected_C, best_k).items():
                model.fit(x_tr, y_all[tr_idx])
                p = model.predict_proba(x_va)
                cols = [cls_idx[c] for c in model.classes_]
                oof[name][np.ix_(va_idx, cols)] = p

        # ---- blend weights by minimising OOF multi-class log loss (SLSQP) ----
        y_idx = np.array([cls_idx[c] for c in y_all])
        rows = np.arange(n)

        def blend_loss(w):
            p = w[0] * oof["lr"] + w[1] * oof["knn"] + w[2] * oof["lda"]
            p = p / p.sum(axis=1, keepdims=True)
            p = np.clip(p, 1e-15, 1.0)
            return float(-np.mean(np.log(p[rows, y_idx])))

        res = minimize(
            blend_loss,
            x0=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        )
        w = res.x
        print("blend_weights lr=%.4f knn=%.4f lda=%.4f oof_log_loss=%.5f"
              % (w[0], w[1], w[2], res.fun))

        tracker.log_params({"selected_C": selected_C, "best_k": best_k, "ensemble": "lr+knn+lda"})
        tracker.log_metric("w_lr", w[0])
        tracker.log_metric("w_knn", w[1])
        tracker.log_metric("w_lda", w[2])
        tracker.log_metric("oof_blend_log_loss", res.fun)

        # ---- refit all three on the full standardised training set, blend, write ----
        proba = np.zeros((x_test_s.shape[0], m))
        for wi, name in zip(w, MODEL_NAMES):
            model = make_models(selected_C, best_k)[name]
            model.fit(x_all_s, y_all)
            p = model.predict_proba(x_test_s)
            full = np.zeros((x_test_s.shape[0], m))
            full[:, [cls_idx[c] for c in model.classes_]] = p
            proba += wi * full
        proba /= proba.sum(axis=1, keepdims=True)

        out = pd.DataFrame(proba, columns=cls_list)
        for c in classes:
            if c not in out.columns:
                out[c] = 0.0
        sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
