"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 1,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine pickup-dropoff distance, passenger count,
and simple calendar features from the pickup timestamp, after dropping training rows
with missing or clearly invalid values. Test rows with invalid coordinates get the
training-mean fare instead of a model prediction. A categorical airport-route
feature identifies trips to and from JFK, LaGuardia, and Newark.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tracking import tracking_run

COMP = "new-york-city-taxi-fare-prediction"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
TRAIN_FILE = "labels.csv"  # mle-bench writes the labelled training rows under this name
N_ROWS = int(os.environ.get("N_ROWS", 1_000_000))  # rows read from the top of the training file
SEED = 0
N_JOBS = 8

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


def airport_id(longitude, latitude):
    """Nearest airport containing an endpoint, or zero for missing/outside points."""
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    ids = np.zeros(longitude.shape, dtype=np.int8)
    nearest = np.full(longitude.shape, np.inf)
    for airport, lon, lat, radius in (
        (1, -73.7781, 40.6413, 2.5),
        (2, -73.8740, 40.7769, 2.0),
        (3, -74.1770, 40.6895, 2.0),
    ):
        distance = haversine_km(longitude, latitude, lon, lat)
        qualifies = np.isfinite(distance) & (distance <= radius) & (distance < nearest)
        ids[qualifies] = airport
        nearest[qualifies] = distance[qualifies]
    return ids


def featurize(df):
    """Baseline features plus one fixed-category airport-route column."""
    t = pd.to_datetime(df["pickup_datetime"], utc=True)
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
    pickup_airport = airport_id(df["pickup_longitude"], df["pickup_latitude"])
    dropoff_airport = airport_id(df["dropoff_longitude"], df["dropoff_latitude"])
    feats["airport_route"] = pd.Categorical(
        4 * pickup_airport + dropoff_airport, categories=range(16)
    )
    return feats


def train(log):
    log("log_params", {
        "n_rows": N_ROWS, "seed": SEED, "n_jobs": N_JOBS,
        "objective": "regression_l1", "metric": "rmse", "n_estimators": 1000, "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8,
        "subsample_freq": 1, "colsample_bytree": 0.8,
        "validation_fraction": 0.1, "early_stopping_rounds": 50,
        "airport_route": True,
    })
    train = pd.read_csv(os.path.join(PUBLIC, TRAIN_FILE), nrows=N_ROWS)
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    print(f"train rows read={len(train)}")
    log("log_metric", "train_rows_read", len(train))
    train = clean(train)
    print(f"train rows after cleaning={len(train)}")
    log("log_metric", "train_rows_clean", len(train))

    x_all = featurize(train)
    y_all = train["fare_amount"].values
    x_test = featurize(test.fillna(0))[x_all.columns]  # test rows are never dropped
    test_valid = valid_mask(test.fillna(0)).values
    mean_fare = float(y_all.mean())
    print(f"test rows={len(test)} with invalid coordinates={int((~test_valid).sum())} -> predicted as mean fare {mean_fare:.3f}")
    log("log_metric", "test_rows", len(test))
    log("log_metric", "invalid_test_rows", int((~test_valid).sum()))
    log("log_metric", "mean_fare", mean_fare)

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y_all, test_size=0.1, random_state=SEED)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="rmse",
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
    started = time.monotonic()

    def progress(env):
        step = env.iteration + 1
        if step % 200 == 0:
            elapsed = time.monotonic() - started
            print(f"training_iteration={step} training_elapsed_seconds={elapsed:.1f}", flush=True)
            log("log_metric", "training_iteration", step, step=step)
            log("log_metric", "training_elapsed_seconds", elapsed, step=step)

    model.fit(
        x_tr, y_tr,
        categorical_feature=["airport_route"],
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), progress],
    )
    print(f"best_iteration={model.best_iteration_}")
    log("log_metric", "best_iteration", model.best_iteration_)

    pred = model.predict(x_test)
    pred = np.where(test_valid, pred, mean_fare)
    sub = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})
    # keep sample_submission's exact key order
    sub = sample[["key"]].merge(sub, on="key", how="left")
    assert sub["fare_amount"].notna().all(), "missing predictions for some test keys"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


def main():
    with tracking_run() as log:
        train(log)


if __name__ == "__main__":
    main()
