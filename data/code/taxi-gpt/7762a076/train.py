"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 1,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine and Manhattan-aligned grid distances,
passenger count, and simple calendar features from the pickup timestamp, after
dropping training rows with missing or clearly invalid values. Test rows with
invalid coordinates get the training-mean fare instead of a model prediction.
Includes endpoint distances to JFK, LaGuardia, and Newark airport anchors.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Tracking:
    """Best-effort attachment to the run configured by the execution environment."""

    def __init__(self):
        self.client = None
        self.context = None
        self.warned = False

    def disable(self, error):
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {error}", flush=True)
            self.warned = True
        self.client = None

    def __enter__(self):
        try:
            import mlflow
            self.context = mlflow.start_run()
            self.context.__enter__()
            self.client = mlflow
        except Exception as error:
            self.disable(error)
        return self

    def log(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as error:
                self.disable(error)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc_value, traceback)
            except Exception as error:
                self.disable(error)
        return False

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


def featurize(df):
    """Raw coordinates, trip/airport distances, passenger count, and pickup calendar."""
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
    x = 6371.0 * np.cos(np.radians(40.75)) * np.radians(
        df["dropoff_longitude"].to_numpy() - df["pickup_longitude"].to_numpy()
    )
    y = 6371.0 * np.radians(
        df["dropoff_latitude"].to_numpy() - df["pickup_latitude"].to_numpy()
    )
    theta = np.radians(29.0)
    feats["manhattan_grid_km"] = (
        np.abs(x * np.cos(theta) - y * np.sin(theta))
        + np.abs(x * np.sin(theta) + y * np.cos(theta))
    ).astype(np.float32)
    for airport, (longitude, latitude) in {
        "jfk": (-73.7781, 40.6413),
        "lga": (-73.8740, 40.7769),
        "ewr": (-74.1745, 40.6895),
    }.items():
        feats[f"pickup_{airport}_km"] = haversine_km(
            df["pickup_longitude"].to_numpy(), df["pickup_latitude"].to_numpy(),
            longitude, latitude,
        ).astype(np.float32)
        feats[f"dropoff_{airport}_km"] = haversine_km(
            df["dropoff_longitude"].to_numpy(), df["dropoff_latitude"].to_numpy(),
            longitude, latitude,
        ).astype(np.float32)
    return feats


def main(tracking):
    train = pd.read_csv(os.path.join(PUBLIC, TRAIN_FILE), nrows=N_ROWS)
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    print(f"train rows read={len(train)}")
    tracking.log("log_metric", "train_rows_read", len(train))
    train = clean(train)
    print(f"train rows after cleaning={len(train)}")

    x_all = featurize(train)
    y_all = train["fare_amount"].values
    x_test = featurize(test.fillna(0))[x_all.columns]  # test rows are never dropped
    test_valid = valid_mask(test.fillna(0)).values
    mean_fare = float(y_all.mean())
    print(f"test rows={len(test)} with invalid coordinates={int((~test_valid).sum())} -> predicted as mean fare {mean_fare:.3f}")
    tracking.log("log_metrics", {
        "train_rows_clean": len(train), "test_rows": len(test),
        "invalid_test_rows": int((~test_valid).sum()), "fallback_mean_fare": mean_fare,
    })

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
    tracking.log("log_params", {
        **model.get_params(), "rows_requested": N_ROWS, "validation_fraction": 0.1,
        "early_stopping_rounds": 50, "manhattan_grid_angle_degrees": 29.0,
        "manhattan_grid_reference_latitude": 40.75,
    })

    def log_progress(env):
        tracking.log("log_metric", "training_iteration", env.iteration + 1,
                     step=env.iteration + 1, synchronous=False)

    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), log_progress],
    )
    print(f"best_iteration={model.best_iteration_}")
    tracking.log("log_metric", "best_iteration", model.best_iteration_)

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


if __name__ == "__main__":
    with Tracking() as tracking:
        main(tracking)
