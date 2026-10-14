"""Baseline for tabular-playground-series-may-2022.

Small SiLU MLP (PyTorch, CPU) on standardized raw numeric columns plus a simple
per-character ordinal expansion of the categorical string column f_27.
No engineered interaction features, no unique-character counts.
Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COMP = "tabular-playground-series-may-2022"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
STR_COL = "f_27"
STR_LEN = 10
SEED = 0
N_JOBS = int(os.environ.get("EV_N_JOBS", "8"))
TORCH_THREADS = int(os.environ.get("EV_TORCH_THREADS", "4"))
EPOCHS = int(os.environ.get("EV_EPOCHS", "60"))
BATCH_SIZE = 4096
LR_MAX, LR_MIN = 1e-2, 1e-5
L2 = 3e-5
TRAIN_TIME_LIMIT_S = 9 * 60


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
    return feats


def main():
    tracker = _Tracker().start()
    params = dict(
        model="mlp_silu",
        hidden="64-64-64-16",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr_max=LR_MAX,
        lr_min=LR_MIN,
        l2=L2,
        optimizer="adam",
        scheduler="cosine_per_epoch",
        seed=SEED,
        torch_threads=TORCH_THREADS,
        holdout_frac=0.1,
        features="numeric + f_27 per-char ordinal (standardized)",
    )
    tracker.log_params(params)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(TORCH_THREADS)

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))

    x_all = featurize(train)
    y_all = train["target"].values
    x_test = featurize(test)[x_all.columns]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
    )

    scaler = StandardScaler().fit(x_tr.values.astype(np.float64))
    xt_tr = torch.tensor(scaler.transform(x_tr.values), dtype=torch.float32)
    xt_val = torch.tensor(scaler.transform(x_val.values), dtype=torch.float32)
    xt_test = torch.tensor(scaler.transform(x_test.values), dtype=torch.float32)
    yt_tr = torch.tensor(y_tr, dtype=torch.float32)

    n_in = xt_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(n_in, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 16), nn.SiLU(),
        nn.Linear(16, 1),
    )
    linear_weights = [m.weight for m in model if isinstance(m, nn.Linear)]
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    def predict(x):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 65536):
                outs.append(torch.sigmoid(model(x[i:i + 65536]).squeeze(1)))
        return torch.cat(outs).numpy()

    best_auc, best_state, best_epoch = -1.0, None, -1
    n_tr = len(xt_tr)
    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        tot_loss, n_batches = 0.0, 0
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            logits = model(xt_tr[idx]).squeeze(1)
            loss = loss_fn(logits, yt_tr[idx])
            l2 = sum((w ** 2).sum() for w in linear_weights)
            total = loss + L2 * l2
            opt.zero_grad()
            total.backward()
            opt.step()
            tot_loss += loss.item()
            n_batches += 1
        sched.step()
        val_auc = roc_auc_score(y_val, predict(xt_val))
        elapsed = time.time() - t_start
        print(f"epoch={epoch + 1} train_loss={tot_loss / n_batches:.5f} val_auc={val_auc:.5f} "
              f"lr={opt.param_groups[0]['lr']:.2e} elapsed={elapsed:.0f}s", flush=True)
        tracker.log_metric("val_auc", val_auc, step=epoch + 1)
        tracker.log_metric("train_loss", tot_loss / n_batches, step=epoch + 1)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if elapsed > TRAIN_TIME_LIMIT_S:
            print(f"time guard hit after epoch {epoch + 1}; stopping training")
            break

    model.load_state_dict(best_state)
    print(f"best_epoch={best_epoch}")
    tracker.log_metric("best_iteration", best_epoch)
    holdout_auc = roc_auc_score(y_val, predict(xt_val))
    print(f"local_holdout_auc={holdout_auc:.5f}")
    tracker.log_metric("local_holdout_auc", holdout_auc)

    proba = predict(xt_test)
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
