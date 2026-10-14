"""Soft-voting ensemble for leaf-classification (replaces the single logistic regression).

The raw 192 shape/margin/texture descriptors feed a sklearn VotingClassifier with soft
voting (uniform weights) over four estimators: (1) standardised linear SVM whose
one-vs-rest decision values are converted to probabilities by Platt-sigmoid calibration
(3-fold stratified CalibratedClassifierCV, class_weight="balanced"), (2) standardised
k-NN whose neighbour count is chosen by stratified CV on log loss, (3) standardised LDA
with automatic shrinkage, and (4) histogram gradient
boosting. Scaling happens inside each pipeline, so the ensemble sees the raw matrix.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import mlflow  # noqa: F401
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def build_ensemble(cv_splits, hgb_validation_fraction):
    svc = Pipeline([
        ("sc", StandardScaler()),
        ("svc", CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                                max_iter=20000, random_state=SEED),
            method="sigmoid",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        )),
    ])
    knn = Pipeline([
        ("sc", StandardScaler()),
        ("knn", GridSearchCV(
            KNeighborsClassifier(weights="distance", metric="minkowski", p=2),
            param_grid={"n_neighbors": list(range(1, 21))},
            scoring="neg_log_loss",
            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED),
            refit=True,
        )),
    ])
    lda = Pipeline([
        ("sc", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    hgb = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.1, max_iter=500, max_leaf_nodes=31,
        min_samples_leaf=10, early_stopping=True,
        validation_fraction=hgb_validation_fraction,
        random_state=SEED,
    )
    return VotingClassifier(
        estimators=[("svc", svc), ("knn", knn), ("lda", lda), ("hgb", hgb)],
        voting="soft", weights=None, n_jobs=-1,
    )


class _MLflow:
    """Best-effort mlflow logging: one failure disables it, never kills the run."""

    def __init__(self):
        self.active = False
        if not _MLFLOW_OK:
            return
        try:
            mlflow.start_run()
            self.active = True
        except Exception as e:
            warnings.warn(f"mlflow start_run failed, continuing untracked: {e}")

    def log_params(self, params):
        if self.active:
            try:
                mlflow.log_params(params)
            except Exception as e:
                warnings.warn(f"mlflow log_params failed: {e}")
                self.active = False

    def log_metric(self, name, value, step=None):
        if self.active:
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as e:
                warnings.warn(f"mlflow log_metric failed: {e}")
                self.active = False

    def end(self):
        if self.active:
            try:
                mlflow.end_run()
            except Exception:
                pass


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    x_all = featurize(train)
    y_all = np.asarray(train["species"].tolist(), dtype=object)  # plain object array: safe for sklearn fancy indexing under pyarrow-backed pandas

    # The spec assumed a 990-row train set (10 samples/class). The actual prepared train
    # split has 891 rows with 6..10 per class and 99 test rows, so the two locked knobs
    # are capped to what stratification on THIS data allows (idea unchanged):
    #  - kNN grid CV: 10-fold stratified needs >=10/class -> use min(10, min class count)
    #  - HGB early stopping: validation_fraction=0.1 -> ~89 val rows < 99 classes; raise
    #    the fraction just enough that the internal stratified split has >=1/class.
    _, class_counts = np.unique(y_all, return_counts=True)
    n_classes = len(class_counts)
    cv_splits = min(10, int(class_counts.min()))
    hgb_validation_fraction = max(0.1, (n_classes + 1) / len(y_all))
    x_test = featurize(test)

    mlf = _MLflow()
    mlf.log_params({
        "model": "soft_voting_svc_knn_lda_hgb",
        "seed": SEED,
        "n_features": x_all.shape[1],
        "svc_C": 1.0,
        "knn_neighbors_grid": "1..20",
        "hgb_learning_rate": 0.1,
        "hgb_max_iter": 500,
    })

    # No local holdout: with only 6..10 samples per class, any holdout split would
    # shrink the per-class counts further and distort the locked CV scheme. The
    # submitted model is fit on ALL training data; telemetry comes from the kNN
    # grid's own stratified CV score (a legitimate train-only estimate).
    mlf.log_params({"cv_splits": cv_splits, "hgb_validation_fraction": hgb_validation_fraction})
    clf = build_ensemble(cv_splits, hgb_validation_fraction)
    clf.fit(x_all, y_all)
    knn_cv = clf.named_estimators_["knn"].named_steps["knn"]
    print("knn_best_n_neighbors=%d knn_cv_neglogloss=%.5f"
          % (knn_cv.best_params_["n_neighbors"], knn_cv.best_score_))
    mlf.log_metric("knn_best_n_neighbors", float(knn_cv.best_params_["n_neighbors"]))
    mlf.log_metric("knn_cv_logloss", -knn_cv.best_score_)

    proba = clf.predict_proba(x_test).astype(np.float64)
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
    mlf.end()


if __name__ == "__main__":
    main()
