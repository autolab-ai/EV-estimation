"""Baseline for dog-breed-identification.

Frozen-backbone transfer learning: a torchvision ResNet-50 with IMAGENET1K_V1
weights is used purely as a feature extractor (no fine-tuning). Every train and
test image is resized/center-cropped to 224px, pushed through the network, and
the 2048-d global-max-pooled activation is kept. A plain sklearn
LogisticRegression (C=0.3) is then fitted on those raw features to predict the
120 breeds.

Deliberately plain: no augmentation, no test-time augmentation, no feature
normalisation, no multi-backbone concatenation, no fine-tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import tempfile
import time
from contextlib import contextmanager


@contextmanager
def worker_tempdir():
    """Keep worker IPC sockets short even when inherited TMPDIR is deeply nested."""
    previous_env = {key: os.environ.get(key) for key in ("TMPDIR", "TEMP", "TMP")}
    previous_tempdir = tempfile.tempdir
    # Ignore inherited long paths when selecting the platform's temporary root.
    for key in previous_env:
        os.environ.pop(key, None)
    tempfile.tempdir = None
    try:
        with tempfile.TemporaryDirectory(prefix="dog-") as path:
            for key in previous_env:
                os.environ[key] = path
            tempfile.tempdir = path
            yield
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = previous_tempdir


class Tracking:
    """Best-effort telemetry: a tracking failure must not stop training."""

    def __init__(self):
        self.client = None
        self.warned = False

    def disable(self, exc):
        self.client = None
        if not self.warned:
            print(f"Warning: MLflow tracking unavailable: {exc}", flush=True)
            self.warned = True

    def log(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as exc:
                self.disable(exc)


@contextmanager
def tracked_run():
    tracking = Tracking()
    context = None
    error_info = (None, None, None)
    try:
        import mlflow
        candidate = mlflow.start_run()
        candidate.__enter__()
        context = candidate
        tracking.client = mlflow
    except Exception as exc:
        tracking.disable(exc)
    try:
        yield tracking
    except BaseException:
        import sys
        error_info = sys.exc_info()
        raise
    finally:
        if context is not None:
            try:
                context.__exit__(*error_info)
            except Exception as exc:
                tracking.disable(exc)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

with open(os.environ.get("DOGBREED_CONFIG", Path(__file__).with_name("config.json"))) as config_file:
    CONFIG = json.load(config_file)
COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", CONFIG["data_dir"])
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", CONFIG["batch_size"]))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", CONFIG["num_workers"]))

TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageFolderList(Dataset):
    """Loads <root>/<id>.jpg for a list of ids, in the given order."""

    def __init__(self, root, ids):
        self.root = root
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            return TFM(img.convert("RGB"))


@torch.no_grad()
def extract(model, root, ids, device):
    loader = DataLoader(
        ImageFolderList(root, ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        feats.append(out.flatten(1).float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def fit_temperature(logits, targets):
    """Fit one scalar using only the internal training-data holdout."""
    rows = np.arange(len(targets))

    def objective(s):
        scaled = logits / np.exp(s)
        return np.mean(logsumexp(scaled, axis=1) - scaled[rows, targets])

    result = minimize_scalar(
        objective, method="bounded", bounds=(np.log(0.5), np.log(3.0)),
        options={"xatol": 0.001, "maxiter": 40},
    )
    return float(np.exp(result.x))


def main(tracking):
    started = time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tracking.log("log_params", {
        "backbone": "resnet50", "weights": "IMAGENET1K_V1",
        "pooling": "global_max",
        "image_size": 224, "resize_size": 256, "seed": SEED,
        "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
        "classifier_C": 0.3, "classifier_max_iter": 1000,
        "validation_fraction": 0.15, "temperature_min": 0.5,
        "temperature_max": 3.0, "temperature_xatol": 0.001,
        "temperature_maxiter": 40,
    })

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=weights)
    net.avgpool = torch.nn.AdaptiveMaxPool2d((1, 1))
    net.fc = torch.nn.Identity()  # 2048-d pooled features
    net.eval().to(device)

    x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
    x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device)
    print(f"features: train={x_train.shape} test={x_test.shape}")

    tracking.log("log_metric", "feature_extraction_seconds", time.monotonic() - started, step=0)
    y_train = labels["breed"].values
    fit_indices, validation_indices = train_test_split(
        np.arange(len(y_train)), test_size=0.15, stratify=y_train, random_state=SEED,
    )
    calibration_clf = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED)
    calibration_clf.fit(x_train[fit_indices], y_train[fit_indices])
    validation_logits = calibration_clf.decision_function(x_train[validation_indices])
    class_indices = {breed: i for i, breed in enumerate(calibration_clf.classes_)}
    validation_targets = np.array([class_indices[breed] for breed in y_train[validation_indices]])
    temperature = fit_temperature(validation_logits, validation_targets)
    print(f"calibrated temperature={temperature:.6f}", flush=True)
    tracking.log("log_metric", "temperature", temperature, step=0)

    # Keep the original classifier recipe and refit on every labeled feature.
    clf = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED)
    clf.fit(x_train, y_train)

    proba = softmax(clf.decision_function(x_test) / temperature, axis=1).astype(np.float64)
    # reorder model classes into the sample-submission column order
    order = [list(clf.classes_).index(b) for b in breeds]
    proba = proba[:, order]
    proba /= proba.sum(axis=1, keepdims=True)  # rows must sum to 1

    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # Graded only by the hill's evaluator; no in-run grading.
    elapsed = time.monotonic() - started
    print(f"submission written in {elapsed:.1f} seconds", flush=True)
    tracking.log("log_metric", "training_seconds", elapsed, step=0)


if __name__ == "__main__":
    with worker_tempdir(), tracked_run() as tracking:
        main(tracking)
