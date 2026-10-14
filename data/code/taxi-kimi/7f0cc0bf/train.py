"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 1,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine pickup-dropoff distance, passenger count,
simple calendar features from the pickup timestamp, and pickup_cell_fare — a smoothed
mean fare of the pickup point's ~100m grid cell computed strictly out-of-fold on the
training data (5-fold) to avoid target leakage — after dropping training rows
with missing or clearly invalid values. Test rows with invalid coordinates get the
training-mean fare instead of a model prediction. No other engineered features,
no airport/landmark handling, no tuning.
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
from sklearn.model_selection import KFold, train_test_split

try:
    import mlflow
    _MLFLOW_OK = True
except Exception as _e:  # tracking is best-effort; never let it kill training
    print(f"mlflow unavailable: {_e}")
    _MLFLOW_OK = False


def mlflow_start():
    """Attach to the preconfigured mlflow run, or no-op if tracking fails."""
    if not _MLFLOW_OK:
        return contextlib.nullcontext()
    try:
        if mlflow.active_run() is not None:
            return contextlib.nullcontext()  # run already active; log to it directly
        return mlflow.start_run()
    except Exception as e:
        print(f"mlflow start_run failed: {e}")
        return contextlib.nullcontext()


class _MLLogger:
    """log_metric/log_params wrappers that disable themselves after one failure."""

    def __init__(self):
        self.ok = _MLFLOW_OK

    def params(self, d):
        if self.ok:
            try:
                mlflow.log_params(d)
            except Exception as e:
                print(f"mlflow log_params failed: {e}")
                self.ok = False

    def metric(self, name, value, step=None):
        if self.ok:
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as e:
                print(f"mlflow log_metric failed: {e}")
                self.ok = False

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
    """Raw coordinates + haversine distance + passenger count + hour / weekday / month / year of pickup."""
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
    return feats


def cell_key(df):
    """Unique integer id for the ~100m pickup grid cell (rounding to 3 decimals)."""
    lat = (df["pickup_latitude"].round(3) * 1000).round().astype(np.int64)
    lon = (df["pickup_longitude"].round(3) * 1000).round().astype(np.int64)
    return lat * 1_000_000 + lon


def smoothed_cell_encoding(cells, y, global_mean, smooth=30):
    """Per-cell mean fare smoothed toward the global mean: (sum + m*mu) / (count + m)."""
    g = pd.DataFrame({"cell": cells, "y": y}).groupby("cell")["y"].agg(["sum", "count"])
    return (g["sum"] + smooth * global_mean) / (g["count"] + smooth)


def oof_cell_fare(train, y, global_mean, n_splits=5, seed=0):
    """Leak-free out-of-fold pickup_cell_fare for training rows: each row's cell
    encoding is computed only from the other folds; unseen cells map to global_mean."""
    cells = cell_key(train)
    oof = np.full(len(train), global_mean, dtype=float)
    for fit_idx, enc_idx in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train):
        enc = smoothed_cell_encoding(cells.iloc[fit_idx].values, y[fit_idx], global_mean)
        oof[enc_idx] = cells.iloc[enc_idx].map(enc).fillna(global_mean).values
    return oof


def main():
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

    # pickup_cell_fare: smoothed mean fare of the ~100m pickup grid cell.
    # Training values are out-of-fold (5-fold) to avoid target leakage; test
    # values use the encoding computed over the entire cleaned training set.
    x_all["pickup_cell_fare"] = oof_cell_fare(train, y_all, mean_fare)
    enc_full = smoothed_cell_encoding(cell_key(train).values, y_all, mean_fare)
    x_test["pickup_cell_fare"] = cell_key(test.fillna(0)).map(enc_full).fillna(mean_fare).values
    print(f"pickup_cell_fare: {train.pipe(cell_key).nunique()} train cells, "
          f"train mean={x_all['pickup_cell_fare'].mean():.3f}, test mean={x_test['pickup_cell_fare'].mean():.3f}")

    print(f"test rows={len(test)} with invalid coordinates={int((~test_valid).sum())} -> predicted as mean fare {mean_fare:.3f}")

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y_all, test_size=0.1, random_state=SEED)

    mllog = _MLLogger()
    mllog.params({
        "n_rows": N_ROWS, "model": "LGBMRegressor", "objective": "huber", "huber_alpha": 1.0,
        "n_estimators": 1000,
        "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED,
        "feature_pickup_cell_fare": True, "cell_fare_smooth": 30, "cell_fare_folds": 5,
    })
    with mlflow_start():
        model = lgb.LGBMRegressor(
            objective="huber",
            alpha=1.0,  # Huber delta: residuals above $1 get linear loss
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
        local_rmse = mean_squared_error(y_val, val_pred) ** 0.5
        print(f"local_holdout_rmse={local_rmse:.5f}")
        mllog.metric("best_iteration", float(model.best_iteration_))
        mllog.metric("local_holdout_rmse", local_rmse)

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
    main()
