"""L1: small PyTorch MLP (64-64-64-16, SiLU) replacing LightGBM for tabular-playground-series-may-2022.

Fully connected network trained on CPU on the raw numeric columns plus a simple
per-character ordinal expansion of the categorical string column f_27.
No engineered interaction features, no unique-character counts, no tuning.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

COMP = "tabular-playground-series-may-2022"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
STR_COL = "f_27"
STR_LEN = 10
SEED = 0
N_JOBS = int(os.environ.get("EV_N_JOBS", "8"))
EPOCHS = int(os.environ.get("EV_EPOCHS", "80"))
WARM_EPOCHS = 20
BATCH_SIZE = 4096
LR_MAX = 0.01
LR_MIN = 1e-5
TIME_GUARD_SEC = 8 * 60


class _Tracker:
    """Best-effort MLflow tracking: never lets telemetry kill training."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self._cm = None

    def start(self):
        try:
            import mlflow

            self.mlflow = mlflow
            self._cm = mlflow.start_run()
            self._cm.__enter__()
            self.ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] disabled: {exc}")
            self.ok = False
        return self

    def log_params(self, params):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(params)
        except Exception as exc:
            print(f"[mlflow] log_params failed, disabling: {exc}")
            self.ok = False

    def log_metric(self, key, value, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            print(f"[mlflow] log_metric failed, disabling: {exc}")
            self.ok = False

    def finish(self):
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass


def featurize(df):
    """Numeric columns as-is + f_27 split into one ordinal column per character."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    for i in range(STR_LEN):
        feats[f"{STR_COL}_{i}"] = chars.str[i].map(ord).astype(np.int16) - ord("A")
    # Raw additive interaction sums (no binning/clipping); trees pick their own splits.
    feats["i_00_01_26"] = (df["f_00"] + df["f_01"] + df["f_26"]).astype(np.float32)
    feats["i_02_21"] = (df["f_02"] + df["f_21"]).astype(np.float32)
    feats["i_05_22"] = (df["f_05"] + df["f_22"]).astype(np.float32)
    return feats


def predict_proba(model, x, chunk=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            out.append(torch.sigmoid(model(x[i:i + chunk])).squeeze(1))
    return torch.cat(out).numpy()


def main():
    t_start = time.time()
    tracker = _Tracker().start()
    params = dict(
        model="mlp",
        hidden="64-64-64-16",
        activation="silu",
        lr=LR_MAX,
        lr_min=LR_MIN,
        weight_decay=6e-5,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        warm_epochs=WARM_EPOCHS,
        seed=SEED,
        n_threads=min(N_JOBS, os.cpu_count()),
        holdout_frac=0.1,
        time_guard_sec=TIME_GUARD_SEC,
        features="numeric + f_27 per-char ordinal + 3 additive sums",
    )
    tracker.log_params(params)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)[x_all.columns]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )

    scaler = StandardScaler().fit(x_tr.values.astype(np.float32))
    xt_tr = torch.from_numpy(scaler.transform(x_tr.values.astype(np.float32)).astype(np.float32))
    xt_val = torch.from_numpy(scaler.transform(x_val.values.astype(np.float32)).astype(np.float32))
    xt_test = torch.from_numpy(scaler.transform(x_test.values.astype(np.float32)).astype(np.float32))
    yt_tr = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(N_JOBS, os.cpu_count()))
    n_in = xt_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(n_in, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 16), nn.SiLU(),
        nn.Linear(16, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX, weight_decay=6e-5)
    gen = torch.Generator().manual_seed(SEED)

    n = xt_tr.shape[0]
    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        if epoch <= WARM_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (
                1 + math.cos(math.pi * (epoch - WARM_EPOCHS) / (EPOCHS - WARM_EPOCHS))
            )
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = xt_tr[idx], yt_tr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.shape[0]
            cnt += idx.shape[0]
        val_pred = predict_proba(model, xt_val)
        auc = roc_auc_score(y_val, val_pred)
        tracker.log_metric("val_auc", auc, step=epoch)
        tracker.log_metric("train_loss", tot / cnt, step=epoch)
        tracker.log_metric("lr", lr, step=epoch)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elapsed = time.time() - t_start
        print(f"epoch={epoch} lr={lr:.6f} loss={tot / cnt:.5f} val_auc={auc:.5f} "
              f"elapsed={elapsed:.0f}s", flush=True)
        if elapsed > TIME_GUARD_SEC:
            print("time guard hit; stopping training", flush=True)
            break

    print(f"best_iteration={best_epoch}")
    tracker.log_metric("best_iteration", best_epoch)
    print(f"local_holdout_auc={best_auc:.5f}")
    tracker.log_metric("local_holdout_auc", best_auc)

    model.load_state_dict(best_state)
    proba = predict_proba(model, xt_test)
    sub = pd.DataFrame({"id": test["id"].values, "target": proba})
    # keep sample_submission's exact id order/dtype
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert sub["target"].notna().all(), "missing predictions for some test ids"
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading
    tracker.finish()


if __name__ == "__main__":
    main()
