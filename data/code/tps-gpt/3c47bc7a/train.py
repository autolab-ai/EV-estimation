"""Bounded CPU MLP on numeric features and the ten f_27 character ordinals.

The training-only holdout selects a checkpoint; only the frozen evaluator
scores the submission. No additional features, ensembling, or full-data refit.
"""

import time

SCRIPT_START = time.monotonic()

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

COMP = "tabular-playground-series-may-2022"
STR_COL = "f_27"
STR_LEN = 10
SEED = 0


class _Tracker:
    """Best-effort MLflow tracking: never lets telemetry kill training."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self._cm = None

    def __enter__(self):
        try:
            import mlflow

            self.mlflow = mlflow
            self._cm = mlflow.start_run()
            self._cm.__enter__()
            self.ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] disabled: {exc}", flush=True)
            self.ok = False
        return self

    def log_params(self, params):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(params)
        except Exception as exc:
            print(f"[mlflow] log_params failed, disabling: {exc}", flush=True)
            self.ok = False

    def log_metric(self, key, value, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            print(f"[mlflow] log_metric failed, disabling: {exc}", flush=True)
            self.ok = False

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc_value, traceback)
            except Exception:
                pass


def load_config():
    path = os.environ.get("EV_CONFIG", str(Path(__file__).with_name("config.json")))
    with open(path) as handle:
        config = json.load(handle)
    config["data_dir"] = os.environ.get("EV_DATA_DIR", config["data_dir"])
    return config


def featurize(df):
    """Numeric columns as-is + f_27 split into one ordinal column per character."""
    feats = df.drop(columns=[c for c in ("id", "target", STR_COL) if c in df.columns])
    chars = df[STR_COL].astype(str)
    for i in range(STR_LEN):
        feats[f"{STR_COL}_{i}"] = chars.str[i].map(ord).astype(np.int16) - ord("A")
    return feats


def build_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.SiLU(),
        nn.Linear(256, 128),
        nn.SiLU(),
        nn.Linear(128, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    ).cpu()


@torch.inference_mode()
def predict_proba(model, inputs, batch_size):
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        logits = model(inputs[start : start + batch_size]).squeeze(1)
        outputs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outputs)


def fit_model(model, x_tr, y_tr, x_val, y_val, config, tracker, deadline):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        min_lr=config["min_lr"],
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    stop_reason = "epoch limit"

    for epoch in range(1, config["max_epochs"] + 1):
        if time.monotonic() >= deadline:
            stop_reason = "time limit"
            break
        model.train()
        # Direct tensor batching shuffles each epoch without worker processes.
        order = torch.randperm(len(x_tr))
        total_loss = 0.0
        completed = True
        for start in range(0, len(x_tr), config["batch_size"]):
            if time.monotonic() >= deadline:
                completed = False
                break
            indices = order[start : start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tr[indices]).squeeze(1)
            loss = criterion(logits, y_tr[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(indices)

        if not completed:
            stop_reason = "time limit; incomplete epoch discarded"
            break

        # AUC is computed only on the 10% holdout from public training rows.
        val_pred = predict_proba(model, x_val, config["inference_batch_size"])
        val_auc = roc_auc_score(y_val, val_pred)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(val_auc)

        metrics = {
            "val_auc": val_auc,
            "train_loss": total_loss / len(x_tr),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.monotonic() - SCRIPT_START,
        }
        for key, value in metrics.items():
            tracker.log_metric(key, value, step=epoch)
        print(
            f"epoch={epoch} "
            + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
            flush=True,
        )
        if stale_epochs >= config["early_stopping_patience"]:
            stop_reason = "early stopping"
            break

    if best_state is None:
        raise RuntimeError("Time budget expired before the first validation checkpoint")
    model.load_state_dict(best_state)
    tracker.log_metric("best_iteration", best_epoch)
    tracker.log_metric("local_holdout_auc", best_auc)
    print(f"stop={stop_reason}; best_iteration={best_epoch}", flush=True)


def main():
    config = load_config()
    deadline = SCRIPT_START + config["time_limit_seconds"]
    torch.set_num_threads(config["torch_num_threads"])
    torch.set_num_interop_threads(config["torch_num_interop_threads"])
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with _Tracker() as tracker:
        tracker.log_params(
            dict(
                model="cpu_mlp",
                hidden_layers="256,128,64",
                activation="SiLU",
                dropout=0,
                optimizer="AdamW",
                loss="BCEWithLogitsLoss",
                seed=SEED,
                holdout_frac=0.1,
                features="numeric + f_27 per-char ordinal",
                scaler="StandardScaler fitted on training partition",
                **config,
            )
        )
        public = Path(config["data_dir"]) / COMP / "prepared" / "public"
        train = pd.read_csv(public / "train.csv")
        test = pd.read_csv(public / "test.csv")
        sample = pd.read_csv(public / "sample_submission.csv")

        x_all = featurize(train)
        y_all = train["target"].values
        x_test = featurize(test)[x_all.columns]
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED
        )

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr.to_numpy(dtype=np.float32))
        x_val = scaler.transform(x_val.to_numpy(dtype=np.float32))
        x_test = scaler.transform(x_test.to_numpy(dtype=np.float32))
        x_tr = torch.from_numpy(np.ascontiguousarray(x_tr, dtype=np.float32))
        x_val = torch.from_numpy(np.ascontiguousarray(x_val, dtype=np.float32))
        x_test = torch.from_numpy(np.ascontiguousarray(x_test, dtype=np.float32))
        y_tr = torch.from_numpy(np.ascontiguousarray(y_tr, dtype=np.float32))

        model = build_model(x_tr.shape[1])
        fit_model(model, x_tr, y_tr, x_val, y_val, config, tracker, deadline)
        proba = predict_proba(model, x_test, config["inference_batch_size"])
        sub = pd.DataFrame({"id": test["id"].values, "target": proba})
        # keep sample_submission's exact id order/dtype
        sub = sample[["id"]].merge(sub, on="id", how="left")
        assert sub["target"].notna().all(), "missing predictions for some test ids"
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
