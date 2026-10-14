"""Baseline + classical computer-vision features for leaf-classification.

The 192 precomputed shape/margin/texture descriptors, augmented with 48 features
computed from the raw binary leaf images (7 Hu moments, 9 contour shape scalars,
32-bin radial-distance histogram), standardised, fed to a plain multinomial
logistic regression. Everything else is identical to the root baseline.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import contextlib

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
IMG_DIR = os.path.join(PUBLIC, "images")
SEED = 0
C = 1.0          # modest, untuned regularisation: lands the root near the leaderboard median

HU_NAMES = [f"hu_{i}" for i in range(1, 8)]
SCALAR_NAMES = [
    "cv_contour_area", "cv_perimeter", "cv_aspect_ratio", "cv_extent",
    "cv_solidity", "cv_major_axis", "cv_minor_axis", "cv_eccentricity",
    "cv_orientation",
]
DIST_NAMES = [f"cv_dist_{i}" for i in range(1, 33)]
IMG_FEATURE_NAMES = HU_NAMES + SCALAR_NAMES + DIST_NAMES  # 7 + 9 + 32 = 48


def image_features(image_id):
    """48 classical CV features for one leaf image; zeros on any failure."""
    try:
        img = cv2.imread(os.path.join(IMG_DIR, f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("unreadable image")
        mask = (img > 127).astype(np.uint8)

        # (a) 7 Hu moments of the mask, log-transformed.
        hu = cv2.HuMoments(cv2.moments(mask)).flatten().astype(np.float64)
        hu_feats = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("no contour")
        cnt = max(contours, key=cv2.contourArea)

        # (b) 9 shape scalars.
        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))
        _, _, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / float(h) if h > 0 else 0.0
        extent = area / float(w * h) if w * h > 0 else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if len(cnt) >= 5:
            (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(cnt)
            major = float(max(axis_a, axis_b))
            minor = float(min(axis_a, axis_b))
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2))) if major > 0 else 0.0
            orientation = float(angle)
        else:
            major = minor = eccentricity = orientation = 0.0
        scalar_feats = [area, perimeter, aspect, extent, solidity,
                        major, minor, eccentricity, orientation]

        # (c) 32-bin histogram of contour-point distances to the centroid.
        pts = cnt.reshape(-1, 2).astype(np.float64)
        centroid = pts.mean(axis=0)
        dists = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
        max_d = dists.max()
        norm = dists / max_d if max_d > 0 else dists
        hist, _ = np.histogram(norm, bins=32, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total

        return np.concatenate([hu_feats, np.asarray(scalar_feats), hist])
    except Exception:
        return np.zeros(48, dtype=np.float64)


def featurize(df):
    base = df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)
    img = np.vstack([image_features(i) for i in df["id"].values])
    return np.hstack([base, img])


def main():
    mlflow = None
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
    except Exception as exc:  # tracking is best-effort, never kill the run
        print("mlflow unavailable, continuing untracked: %s" % exc)

    run_ctx = contextlib.nullcontext()
    if mlflow is not None:
        try:
            run_ctx = mlflow.start_run()
        except Exception as exc:
            print("mlflow start_run failed, continuing untracked: %s" % exc)
            mlflow = None

    with run_ctx:
        if mlflow is not None:
            try:
                mlflow.log_params({"C": C, "seed": SEED, "n_base_features": 192,
                                   "n_image_features": 48, "model": "logistic_regression"})
            except Exception:
                pass

        train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
        test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
        sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
        classes = [c for c in sample.columns if c != "id"]

        x_all, y_all = featurize(train), train["species"].to_numpy(dtype=str)
        x_test = featurize(test)

        scaler = StandardScaler().fit(x_all)
        x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all_s, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        clf.fit(x_tr, y_tr)
        holdout = log_loss(y_val, clf.predict_proba(x_val), labels=clf.classes_)
        print("local_holdout_logloss=%.5f" % holdout)
        if mlflow is not None:
            try:
                mlflow.log_metric("local_holdout_logloss", holdout)
            except Exception:
                pass

        clf.fit(x_all_s, y_all)
        proba = clf.predict_proba(x_test_s).astype(np.float64)
        proba /= proba.sum(axis=1, keepdims=True)
        out = pd.DataFrame(proba, columns=list(clf.classes_))
        for c in classes:
            if c not in out.columns:
                out[c] = 0.0
        sub = pd.concat([pd.DataFrame({"id": test["id"].values}), out[classes]], axis=1)
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)   # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
