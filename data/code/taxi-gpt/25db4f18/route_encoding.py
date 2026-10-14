"""Smoothed directional route fares, cross-fitted within the training split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

CELL_SIZE = 0.01
SMOOTHING = 50.0
N_FOLDS = 5
FOLD_SEED = 0
ROUTE_COLUMNS = [
    "pickup_lon_cell", "pickup_lat_cell", "dropoff_lon_cell", "dropoff_lat_cell"
]


def route_cells(features):
    """Keep pickup and dropoff positions distinct in the directional key."""
    return pd.DataFrame({
        "pickup_lon_cell": np.floor((features["pickup_longitude"] + 75.0) / CELL_SIZE).astype(np.int64),
        "pickup_lat_cell": np.floor((features["pickup_latitude"] - 40.0) / CELL_SIZE).astype(np.int64),
        "dropoff_lon_cell": np.floor((features["dropoff_longitude"] + 75.0) / CELL_SIZE).astype(np.int64),
        "dropoff_lat_cell": np.floor((features["dropoff_latitude"] - 40.0) / CELL_SIZE).astype(np.int64),
    }, index=features.index)


def route_statistics(cells, fares):
    """Aggregate only the supplied fitting rows; targets align by position."""
    rows = cells.copy()
    rows["fare_amount"] = np.asarray(fares, dtype=np.float64)
    stats = rows.groupby(ROUTE_COLUMNS, sort=False)["fare_amount"].agg(S="sum", N="size")
    return stats, float(np.mean(fares))


def encode_routes(cells, stats, mean_fare):
    keys = pd.MultiIndex.from_frame(cells[ROUTE_COLUMNS])
    matched = stats.reindex(keys).fillna(0.0)
    values = (matched["S"] + SMOOTHING * mean_fare) / (matched["N"] + SMOOTHING)
    return values.to_numpy(dtype=np.float32)


def add_route_mean_fare(x_train, y_train, x_validation, x_test):
    """Append just one feature; validation and test never supply targets."""
    train_cells = route_cells(x_train)
    fares = np.asarray(y_train)
    out_of_fold = np.empty(len(x_train), dtype=np.float32)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fit_positions, held_positions in folds.split(x_train):
        stats, mean_fare = route_statistics(train_cells.iloc[fit_positions], fares[fit_positions])
        out_of_fold[held_positions] = encode_routes(train_cells.iloc[held_positions], stats, mean_fare)

    stats, mean_fare = route_statistics(train_cells, fares)
    train_features = x_train.copy()
    validation_features = x_validation.copy()
    test_features = x_test.copy()
    train_features["route_mean_fare"] = out_of_fold
    validation_features["route_mean_fare"] = encode_routes(route_cells(x_validation), stats, mean_fare)
    test_features["route_mean_fare"] = encode_routes(route_cells(x_test), stats, mean_fare)
    return train_features, validation_features, test_features
