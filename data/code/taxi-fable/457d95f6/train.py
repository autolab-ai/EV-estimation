"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 1,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine pickup-dropoff distance, passenger count,
and simple calendar features from the pickup timestamp, after dropping training rows
with missing values or absurd fares. Coordinates outside the NYC box are treated as
missing (NaN) features, so LightGBM learns how to handle them; every test row is
predicted by the model. No other engineered features, no airport/landmark handling, no tuning.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import contextlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

COMP = "new-york-city-taxi-fare-prediction"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
TRAIN_FILE = "labels.csv"  # mle-bench writes the labelled training rows under this name
N_ROWS = int(os.environ.get("N_ROWS", 1_000_000))  # rows read from the top of the training file
SEED = 0
N_JOBS = 8

# NYC bounding box: coordinates outside it are treated as missing (NaN)
LON_MIN, LON_MAX = -75.0, -72.5
LAT_MIN, LAT_MAX = 40.0, 42.0


class Tracker:
    """Best-effort MLflow logging: disables itself after the first failure."""

    def __init__(self):
        self.ok = True
        try:
            import mlflow  # noqa: F401
            self.mlflow = mlflow
        except Exception as e:  # pragma: no cover
            print(f"mlflow unavailable: {e}")
            self.ok = False

    def start_run(self):
        if not self.ok:
            return contextlib.nullcontext()
        try:
            return self.mlflow.start_run()
        except Exception as e:
            print(f"mlflow start_run failed, continuing untracked: {e}")
            self.ok = False
            return contextlib.nullcontext()

    def log_params(self, d):
        if self.ok:
            try:
                self.mlflow.log_params(d)
            except Exception as e:
                print(f"mlflow log_params failed, disabling tracking: {e}")
                self.ok = False

    def log_metric(self, k, v, step=None):
        if self.ok:
            try:
                self.mlflow.log_metric(k, v, step=step)
            except Exception as e:
                print(f"mlflow log_metric failed, disabling tracking: {e}")
                self.ok = False


def point_valid(lon, lat):
    """True where a (lon, lat) point lies inside the NYC box (NaN -> False)."""
    return (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)


def clean(df):
    """Drop training rows with missing values or non-positive / absurd fares.
    Rows with invalid coordinates or passenger counts are kept (handled as NaN features)."""
    df = df.dropna()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 500)]
    return df


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def featurize(df):
    """Raw coordinates + haversine distance + passenger count + hour / weekday / month / year of pickup."""
    t = pd.to_datetime(df["pickup_datetime"], utc=True)
    plon = df["pickup_longitude"].values.astype(float)
    plat = df["pickup_latitude"].values.astype(float)
    dlon = df["dropoff_longitude"].values.astype(float)
    dlat = df["dropoff_latitude"].values.astype(float)
    dist = haversine_km(plon, plat, dlon, dlat)
    # out-of-box (or missing) coordinates become NaN so LightGBM treats them as missing
    with np.errstate(invalid="ignore"):
        pv = point_valid(plon, plat)
        dv = point_valid(dlon, dlat)
    plon = np.where(pv, plon, np.nan)
    plat = np.where(pv, plat, np.nan)
    dlon = np.where(dv, dlon, np.nan)
    dlat = np.where(dv, dlat, np.nan)
    dist = np.where(pv & dv, dist, np.nan)
    feats = pd.DataFrame({
        "pickup_longitude": plon,
        "pickup_latitude": plat,
        "dropoff_longitude": dlon,
        "dropoff_latitude": dlat,
        "distance_km": dist,
        "passenger_count": np.clip(df["passenger_count"].values.astype(float), 0, 6),
        "hour": t.dt.hour.values,
        "weekday": t.dt.weekday.values,
        "month": t.dt.month.values,
        "year": t.dt.year.values,
    }, index=df.index)
    return feats


def main():
    tracker = Tracker()
    with tracker.start_run():
        run(tracker)


def run(tracker):
    train = pd.read_csv(os.path.join(PUBLIC, TRAIN_FILE), nrows=N_ROWS)
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    print(f"train rows read={len(train)}")
    train = clean(train)
    print(f"train rows after cleaning={len(train)}")
    tracker.log_params({"n_rows": N_ROWS, "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63,
                        "min_child_samples": 100, "invalid_coords": "nan_features"})
    tracker.log_metric("train_rows", len(train))

    x_all = featurize(train)
    y_all = train["fare_amount"].values
    x_test = featurize(test)[x_all.columns]  # test rows are never dropped; bad coords -> NaN features
    print(f"test rows={len(test)} with missing/invalid coordinate features={int(x_test['distance_km'].isna().sum())}")

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y_all, test_size=0.1, random_state=SEED)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )
    print(f"best_iteration={model.best_iteration_}")
    val_pred = model.predict(x_val)
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
    print(f"local_holdout_rmse={val_rmse:.5f}")
    tracker.log_metric("best_iteration", model.best_iteration_)
    tracker.log_metric("local_holdout_rmse", val_rmse)

    pred = model.predict(x_test)
    sub = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})
    # keep sample_submission's exact key order
    sub = sample[["key"]].merge(sub, on="key", how="left")
    assert sub["fare_amount"].notna().all(), "missing predictions for some test keys"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
