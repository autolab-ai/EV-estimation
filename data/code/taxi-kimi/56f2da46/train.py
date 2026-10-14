"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 1,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine pickup-dropoff distance, passenger count,
and simple calendar features from the pickup timestamp, after dropping training rows
with missing or clearly invalid values. Test rows with invalid coordinates are
predicted by a small second LightGBM model (same hyperparameters) that uses only
non-coordinate features (passenger count and the pickup-time calendar features),
instead of the flat training-mean fare. No other engineered features,
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

COMP = "new-york-city-taxi-fare-prediction"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
TRAIN_FILE = "labels.csv"  # mle-bench writes the labelled training rows under this name
N_ROWS = int(os.environ.get("N_ROWS", 1_000_000))  # rows read from the top of the training file
SEED = 0
N_JOBS = 8

# features the fallback model may use (everything except the raw coordinates and distance)
FB_FEATURES = ["passenger_count", "hour", "weekday", "month", "year"]

# one shared hyperparameter set so the main model and the fallback model are identical
LGBM_PARAMS = dict(
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


def _mlflow_run():
    """Best-effort MLflow attachment: returns a context manager; falls back to a
    no-op one if tracking is unavailable so a tracking failure can never kill training."""
    import contextlib
    try:
        import mlflow
        return mlflow.start_run()
    except Exception as e:  # noqa: BLE001
        print(f"mlflow disabled: {e}")
        return contextlib.nullcontext()


def _log_metric(name, value):
    try:
        import mlflow
        mlflow.log_metric(name, float(value))
    except Exception:  # noqa: BLE001
        pass


def _log_params(params):
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception:  # noqa: BLE001
        pass

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


def main():
    train = pd.read_csv(os.path.join(PUBLIC, TRAIN_FILE), nrows=N_ROWS)
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    print(f"train rows read={len(train)}")
    train = clean(train)
    print(f"train rows after cleaning={len(train)}")

    x_all = featurize(train)
    y_raw = train["fare_amount"].values  # raw dollar fares, kept for dollar-space reporting
    y_all = np.log1p(y_raw)  # train both models on the log-transformed target
    x_test = featurize(test.fillna(0))[x_all.columns]  # test rows are never dropped
    test_valid = valid_mask(test.fillna(0)).values
    mean_fare = float(y_raw.mean())
    print(f"test rows={len(test)} with invalid coordinates={int((~test_valid).sum())} -> predicted as mean fare {mean_fare:.3f}")

    x_tr, x_val, y_tr, y_val, _, y_val_raw = train_test_split(
        x_all, y_all, y_raw, test_size=0.1, random_state=SEED
    )

    with _mlflow_run():
        _log_params({"n_rows": N_ROWS, "model": "LGBMRegressor", "fallback": "lgbm_time_passenger",
                     **{k: v for k, v in LGBM_PARAMS.items() if k != "verbose"}})

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
        )
        print(f"best_iteration={model.best_iteration_}")
        val_pred = np.expm1(model.predict(x_val))  # back to dollar space for reporting
        holdout_rmse = mean_squared_error(y_val_raw, val_pred) ** 0.5
        print(f"local_holdout_rmse={holdout_rmse:.5f}")
        _log_metric("local_holdout_rmse", holdout_rmse)
        _log_metric("best_iteration", model.best_iteration_)

        # Fallback model for test rows with invalid coordinates: same hyperparameters,
        # trained on the same split, but only on non-coordinate features (passenger
        # count + pickup-time calendar features) instead of a flat mean fare.
        fallback = lgb.LGBMRegressor(**LGBM_PARAMS)
        fallback.fit(
            x_tr[FB_FEATURES], y_tr,
            eval_set=[(x_val[FB_FEATURES], y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
        )
        fb_val_pred = np.expm1(fallback.predict(x_val[FB_FEATURES]))
        fb_rmse = mean_squared_error(y_val_raw, fb_val_pred) ** 0.5
        print(f"fallback_holdout_rmse={fb_rmse:.5f} (vs mean-fare fallback)")
        _log_metric("fallback_holdout_rmse", fb_rmse)

        pred = np.expm1(model.predict(x_test))
        fb_pred = np.expm1(fallback.predict(x_test[FB_FEATURES]))
        pred = np.clip(pred, 2.50, None)  # floor at the historical NYC base fare
        fb_pred = np.clip(fb_pred, 2.50, None)
        pred = np.where(test_valid, pred, fb_pred)
        print(f"fallback applied to {int((~test_valid).sum())} test rows")
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
