"""Baseline for new-york-city-taxi-fare-prediction.

LightGBM regression on a 1,000,000-row sample of the training data using the raw
pickup/dropoff coordinates, the haversine pickup-dropoff distance, passenger count,
and simple calendar features from the pickup timestamp, after dropping training rows
with missing or clearly invalid values. Test rows with invalid coordinates get the
training-mean fare instead of a model prediction. No other engineered features,
no airport/landmark handling, no tuning. This node adds six out-of-fold smoothed
target-encoding features (mean fare and log count) for ~1 km pickup/dropoff grid cells
and ~2 km pickup->dropoff route keys.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

try:
    import mlflow
except Exception:  # pragma: no cover - tracking is optional
    mlflow = None


class _Tracker:
    """Best-effort MLflow wrapper: disables itself after the first failure."""

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

    def log_params(self, d):
        if self.ok:
            try:
                mlflow.log_params(d)
            except Exception as e:
                print(f"mlflow disabled: {e}")
                self.ok = False

    def log_metric(self, k, v, step=None):
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
N_ROWS = int(os.environ.get("N_ROWS", 1_000_000))  # rows read from the top of the training file
SEED = 0
N_JOBS = 8
TE_SMOOTHING = 20.0  # pseudo-count m for smoothed target means
TE_FOLDS = 5
CELL_DEG = 0.01  # ~1 km endpoint cells
ROUTE_DEG = 0.02  # coarser cells for pickup->dropoff route keys

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


def cell_ids(df):
    """Integer ids for pickup cell, dropoff cell (CELL_DEG) and the pickup->dropoff route (ROUTE_DEG)."""
    def grid(lat, lon, deg):
        gi = np.floor(lat.values / deg).astype(np.int64)
        gj = np.floor(lon.values / deg).astype(np.int64)
        return gi * 100_000 + gj  # lat idx ~4000, lon idx ~-7400 -> unique combination

    pickup_cell = grid(df["pickup_latitude"], df["pickup_longitude"], CELL_DEG)
    dropoff_cell = grid(df["dropoff_latitude"], df["dropoff_longitude"], CELL_DEG)
    p_route = grid(df["pickup_latitude"], df["pickup_longitude"], ROUTE_DEG)
    d_route = grid(df["dropoff_latitude"], df["dropoff_longitude"], ROUTE_DEG)
    route_key = p_route * 10_000_000_000 + d_route
    return pd.DataFrame({"pickup_cell": pickup_cell, "dropoff_cell": dropoff_cell, "route_key": route_key},
                        index=df.index)


def te_stats(keys, fares):
    """Per-key sum and count of fare plus the global mean of the statistics rows."""
    g = pd.DataFrame({"k": keys, "y": fares}).groupby("k")["y"].agg(["sum", "count"])
    return g, float(np.mean(fares))


def te_apply(keys, stats, global_mean, m=TE_SMOOTHING):
    """Smoothed mean (sum + m*global)/(count + m) and log1p(count) for each key; unseen keys -> global mean."""
    s = pd.Series(keys).map(stats["sum"]).fillna(0.0).values
    c = pd.Series(keys).map(stats["count"]).fillna(0.0).values
    return (s + m * global_mean) / (c + m), np.log1p(c)


def target_encode(cells_tr, y_tr, cells_other):
    """Out-of-fold target encodings for training rows; full-train statistics for other rows.

    Returns (frame for training rows, list of frames for each other set), each with
    <key>_fare_mean and <key>_log_count columns for route_key, pickup_cell, dropoff_cell.
    """
    names = {"route_key": "route", "pickup_cell": "pickup_cell", "dropoff_cell": "dropoff_cell"}
    out_tr = pd.DataFrame(index=cells_tr.index)
    out_other = [pd.DataFrame(index=c.index) for c in cells_other]
    kf = KFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(cells_tr))
    for col, name in names.items():
        keys_tr = cells_tr[col].values
        mean_col = np.zeros(len(keys_tr))
        cnt_col = np.zeros(len(keys_tr))
        for fit_idx, enc_idx in folds:
            stats, gm = te_stats(keys_tr[fit_idx], y_tr[fit_idx])
            mean_col[enc_idx], cnt_col[enc_idx] = te_apply(keys_tr[enc_idx], stats, gm)
        out_tr[f"{name}_fare_mean"] = mean_col
        out_tr[f"{name}_log_count"] = cnt_col
        stats, gm = te_stats(keys_tr, y_tr)
        for c, o in zip(cells_other, out_other):
            o[f"{name}_fare_mean"], o[f"{name}_log_count"] = te_apply(c[col].values, stats, gm)
    return out_tr, out_other


def main():
    TRACK.start()
    TRACK.log_params({"n_rows": N_ROWS, "te_smoothing": TE_SMOOTHING, "te_folds": TE_FOLDS,
                      "cell_deg": CELL_DEG, "route_deg": ROUTE_DEG, "learning_rate": 0.05,
                      "num_leaves": 63, "n_estimators": 1000, "target_transform": "log1p"})
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

    # out-of-fold target encodings of route / endpoint grid cells (no row sees its own fare)
    test_f = test.fillna(0)
    cells_tr = cell_ids(train.loc[x_tr.index])
    cells_val = cell_ids(train.loc[x_val.index])
    cells_test = cell_ids(test_f)
    te_tr, (te_val, te_test) = target_encode(cells_tr, y_tr, [cells_val, cells_test])
    x_tr = pd.concat([x_tr, te_tr], axis=1)
    x_val = pd.concat([x_val, te_val], axis=1)
    x_test = pd.concat([x_test, te_test], axis=1)[x_tr.columns]
    print(f"features={list(x_tr.columns)}")
    print(f"unique route keys in train={cells_tr['route_key'].nunique()}")

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
    # fit on the log scale: stabilises variance of the right-skewed fare, keeps predictions positive
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)
    model.fit(
        x_tr, y_tr_log,
        eval_set=[(x_val, y_val_log)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )
    print(f"best_iteration={model.best_iteration_}")
    val_pred_log = model.predict(x_val)
    val_rmse_log = mean_squared_error(y_val_log, val_pred_log) ** 0.5
    val_pred = np.expm1(val_pred_log)
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
    print(f"local_holdout_rmse_log={val_rmse_log:.5f}")
    print(f"local_holdout_rmse={val_rmse:.5f}")
    TRACK.log_metric("best_iteration", model.best_iteration_)
    TRACK.log_metric("local_holdout_rmse_log", val_rmse_log)
    TRACK.log_metric("local_holdout_rmse", val_rmse)

    pred = np.expm1(model.predict(x_test))
    pred = np.where(test_valid, pred, mean_fare)
    sub = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})
    # keep sample_submission's exact key order
    sub = sample[["key"]].merge(sub, on="key", how="left")
    assert sub["fare_amount"].notna().all(), "missing predictions for some test keys"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading
    TRACK.end()


if __name__ == "__main__":
    main()
