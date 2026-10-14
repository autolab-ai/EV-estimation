"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 4,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine pickup-dropoff distance, passenger count,
and simple calendar features from the pickup timestamp, after dropping training rows
with missing or clearly invalid values. Test rows with invalid coordinates get the
training-mean fare instead of a model prediction. No other engineered features,
no airport/landmark handling, no tuning.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class _Tracker:
    """Best-effort MLflow logging: disables itself after the first failure."""

    def __init__(self):
        self.ok = mlflow is not None
        self.run = None

    def start(self):
        if not self.ok:
            return
        try:
            self.run = mlflow.start_run()
        except Exception as e:
            print(f"mlflow disabled: {e}")
            self.ok = False

    def params(self, d):
        if self.ok:
            try:
                mlflow.log_params(d)
            except Exception as e:
                print(f"mlflow disabled: {e}")
                self.ok = False

    def metric(self, k, v, step=None):
        if self.ok:
            try:
                mlflow.log_metric(k, float(v), step=step)
            except Exception as e:
                print(f"mlflow disabled: {e}")
                self.ok = False

    def end(self):
        if self.ok and self.run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


TRACK = _Tracker()

COMP = "new-york-city-taxi-fare-prediction"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
TRAIN_FILE = "labels.csv"  # mle-bench writes the labelled training rows under this name
N_ROWS = int(os.environ.get("N_ROWS", 4_000_000))  # rows read from the top of the training file
SEED = 0
N_JOBS = int(os.environ.get("N_JOBS", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)))

# NYC bounding box used only to drop corrupt coordinates
LON_MIN, LON_MAX = -75.0, -72.5
LAT_MIN, LAT_MAX = 40.0, 42.0


def valid_mask(df):
    """Rows whose coordinates are inside the NYC box and passenger count is 1..6."""
    m = df["pickup_longitude"].between(LON_MIN, LON_MAX) & df["dropoff_longitude"].between(LON_MIN, LON_MAX)
    m &= df["pickup_latitude"].between(LAT_MIN, LAT_MAX) & df["dropoff_latitude"].between(LAT_MIN, LAT_MAX)
    m &= df["passenger_count"].between(1, 6)
    return m


def clean(df):
    """Drop training rows with missing values, non-positive or absurd fares, or an
    invalid coordinate / passenger count."""
    df = df.dropna()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 500)]
    return df[valid_mask(df)]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


AIRPORTS = {  # (latitude, longitude)
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
}


def featurize(df):
    """Raw coordinates + haversine distance + passenger count + hour / weekday / month / year of pickup
    + great-circle distance (km) from pickup and dropoff to JFK, LaGuardia and Newark airports."""
    t = pd.to_datetime(df["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", utc=True)
    feats = pd.DataFrame({
        "pickup_longitude": df["pickup_longitude"].values,
        "pickup_latitude": df["pickup_latitude"].values,
        "dropoff_longitude": df["dropoff_longitude"].values,
        "dropoff_latitude": df["dropoff_latitude"].values,
        "distance_km": haversine_km(df["pickup_longitude"].values, df["pickup_latitude"].values,
                                    df["dropoff_longitude"].values, df["dropoff_latitude"].values),
        "passenger_count": df["passenger_count"].values,
        "hour": t.dt.hour.values,
        "weekday": t.dt.weekday.values,
        "month": t.dt.month.values,
        "year": t.dt.year.values,
    }, index=df.index)
    for name, (alat, alon) in AIRPORTS.items():
        for end in ("pickup", "dropoff"):
            feats[f"{end}_{name}_km"] = haversine_km(df[f"{end}_longitude"].values, df[f"{end}_latitude"].values,
                                                     alon, alat)
    return feats


def main():
    TRACK.start()
    TRACK.params({"n_rows": N_ROWS, "n_jobs": N_JOBS, "n_estimators": 1000, "learning_rate": 0.05,
                  "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8})
    train = pd.read_csv(os.path.join(PUBLIC, TRAIN_FILE), nrows=N_ROWS)
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    print(f"train rows read={len(train)}")
    train = clean(train)
    print(f"train rows after cleaning={len(train)}")

    x_all = featurize(train)
    y_all = train["fare_amount"].values
    x_test = featurize(test.fillna(0))[x_all.columns]  # test rows are never dropped
    test_valid = valid_mask(test.fillna(0)).values
    mean_fare = float(y_all.mean())
    print(f"test rows={len(test)} with invalid coordinates={int((~test_valid).sum())} -> predicted as mean fare {mean_fare:.3f}")

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
    holdout_rmse = mean_squared_error(y_val, val_pred) ** 0.5
    print(f"local_holdout_rmse={holdout_rmse:.5f}")
    TRACK.metric("best_iteration", model.best_iteration_)
    TRACK.metric("local_holdout_rmse", holdout_rmse)
    TRACK.metric("train_rows_clean", len(train))
    for i, v in enumerate(model.evals_result_["valid_0"]["rmse"]):
        if i % 50 == 0 or i == len(model.evals_result_["valid_0"]["rmse"]) - 1:
            TRACK.metric("valid_rmse", v, step=i + 1)

    pred = model.predict(x_test)
    pred = np.where(test_valid, pred, mean_fare)
    sub = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})
    # keep sample_submission's exact key order
    sub = sample[["key"]].merge(sub, on="key", how="left")
    assert sub["fare_amount"].notna().all(), "missing predictions for some test keys"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    TRACK.end()
    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
