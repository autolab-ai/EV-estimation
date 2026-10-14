"""Baseline for leaf-classification.

The 192 precomputed shape/margin/texture descriptors, standardised, fed to a plain
multinomial logistic regression. Targets the historical leaderboard median (~0.11
multi-class log loss); deliberately NOT state of the art: no feature selection, no
image features, no ensembling, no calibration, and the regularisation is left at a
single fixed value rather than tuned.

Writes submission/submission.csv, which the hill's evaluator grades.
"""

import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 1.0          # modest, untuned regularisation: lands the root near the leaderboard median
N_FOLDS = 5      # stratified CV folds; fold models are averaged for the submission


def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


# --- geometric descriptors from the binary leaf images (scikit-image) -----------------
GEOM_COLS = [
    "img_h", "img_w", "area", "perimeter", "area_convex", "solidity", "extent",
    "eccentricity", "axis_major", "axis_minor", "axis_ratio", "equiv_diam", "circularity",
    "euler", "n_components", "bbox_h", "bbox_w", "bbox_ratio", "hull_perim_ratio",
    "leaf_fraction",
] + ["hu%d" % i for i in range(7)]
LOG_COLS = ["img_h", "img_w", "area", "perimeter", "area_convex", "axis_major", "axis_minor",
            "equiv_diam", "bbox_h", "bbox_w"]


def _attr(r, new, old):
    return getattr(r, new) if hasattr(r, new) else getattr(r, old)


def image_features(img_id):
    img = Image.open(os.path.join(PUBLIC, "images", f"{img_id}.jpg")).convert("L")
    arr = np.asarray(img)
    mask = arr > 127
    if mask.mean() > 0.5:
        mask = ~mask
    h, w = mask.shape
    lab = measure.label(mask)
    regions = measure.regionprops(lab)
    r = max(regions, key=lambda x: x.area)
    area = float(r.area)
    perim = float(r.perimeter)
    area_convex = float(_attr(r, "area_convex", "convex_area"))
    a_major = float(_attr(r, "axis_major_length", "major_axis_length"))
    a_minor = float(_attr(r, "axis_minor_length", "minor_axis_length"))
    eq_diam = float(_attr(r, "equivalent_diameter_area", "equivalent_diameter"))
    euler = float(measure.regionprops(mask.astype(int))[0].euler_number)
    minr, minc, maxr, maxc = r.bbox
    bh, bw = float(maxr - minr), float(maxc - minc)
    hull = _attr(r, "image_convex", "convex_image").astype(int)
    hull_perim = float(measure.regionprops(hull)[0].perimeter)
    hu = np.asarray(r.moments_hu, dtype=np.float64)
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    feats = [
        float(h), float(w), area, perim, area_convex, float(r.solidity), float(r.extent),
        float(r.eccentricity), a_major, a_minor, a_minor / a_major, eq_diam,
        4.0 * np.pi * area / perim ** 2, euler, float(lab.max()), bh, bw,
        min(bh, bw) / max(bh, bw), hull_perim / perim, area / (h * w),
    ] + list(hu)
    return feats


def geometric_features(ids):
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in ids:
            rows.append(image_features(int(i)))
    x = np.asarray(rows, dtype=np.float64)
    log_idx = [GEOM_COLS.index(c) for c in LOG_COLS]
    x[:, log_idx] = np.log1p(x[:, log_idx])
    x[~np.isfinite(x)] = 0.0
    return x


class _Tracker:
    """Best-effort MLflow logging; disables itself on the first failure."""

    def __init__(self):
        self.ok = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.run = mlflow.start_run()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print("mlflow disabled: %s" % e)

    def log_params(self, d):
        if self.ok:
            try:
                self.mlflow.log_params(d)
            except Exception as e:  # noqa: BLE001
                print("mlflow disabled: %s" % e); self.ok = False

    def log_metric(self, k, v, step=0):
        if self.ok:
            try:
                self.mlflow.log_metric(k, float(v), step=step)
            except Exception as e:  # noqa: BLE001
                print("mlflow disabled: %s" % e); self.ok = False

    def end(self):
        if self.ok:
            try:
                self.mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


def main():
    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]

    tracker = _Tracker()
    tracker.log_params({"C": C, "seed": SEED, "n_geom_features": len(GEOM_COLS), "model": "logreg+geom", "n_folds": N_FOLDS, "protocol": "5fold_oof_avg"})

    x_all, y_all = featurize(train), np.asarray(train["species"].astype(str).tolist(), dtype=object)
    x_test = featurize(test)
    g_all = geometric_features(train["id"].values)
    g_test = geometric_features(test["id"].values)
    x_all = np.hstack([x_all, g_all])
    x_test = np.hstack([x_test, g_test])
    print("feature_matrix_shape=%s" % (x_all.shape,))

    scaler = StandardScaler().fit(x_all)
    x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

    # 5-fold stratified CV: out-of-fold probabilities give the validation number,
    # the five fold models averaged give the final test probabilities (no full refit).
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_classes = None
    oof = np.zeros((x_all_s.shape[0], len(classes)), dtype=np.float64)
    test_proba_sum = np.zeros((x_test_s.shape[0], len(classes)), dtype=np.float64)
    for k, (tr_idx, va_idx) in enumerate(skf.split(x_all_s, y_all)):
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        clf.fit(x_all_s[tr_idx], y_all[tr_idx])
        if fold_classes is None:
            fold_classes = list(clf.classes_)
        assert list(clf.classes_) == fold_classes, "classes differ across folds"
        oof[va_idx] = clf.predict_proba(x_all_s[va_idx])
        test_proba_sum += clf.predict_proba(x_test_s)
        fold_ll = log_loss(y_all[va_idx], oof[va_idx], labels=fold_classes)
        print("fold=%d val_logloss=%.5f" % (k, fold_ll))
        tracker.log_metric("fold_logloss", fold_ll, step=k)
    val_ll = log_loss(y_all, oof, labels=fold_classes)
    print("local_holdout_logloss=%.5f (5-fold OOF over %d rows)" % (val_ll, len(y_all)))
    tracker.log_metric("local_holdout_logloss", val_ll)

    proba = test_proba_sum / N_FOLDS
    proba /= proba.sum(axis=1, keepdims=True)
    out = pd.DataFrame(proba, columns=fold_classes)
    for c in classes:
        if c not in out.columns:
            out[c] = 0.0
    sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
    os.makedirs("submission", exist_ok=True)   # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    tracker.end()


if __name__ == "__main__":
    main()
