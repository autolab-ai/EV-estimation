"""Baseline for dog-breed-identification.

Frozen-backbone transfer learning: a torchvision ResNet-50 with IMAGENET1K_V1
weights is used purely as a feature extractor (no fine-tuning). Every train and
test image is resized to 219px/center-cropped to 192px, pushed through the network, and
the 2048-d global-average-pooled activation is kept. A plain sklearn
LogisticRegression (C=0.3) is then fitted on those raw features to predict the
120 breeds.

Training stays single-view. Test inference averages probabilities from the
original center crop and its horizontal mirror, with sequential backbone passes.
No feature normalisation, multi-backbone concatenation, or fine-tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
from pathlib import Path
import tempfile
import time
from contextlib import contextmanager

with Path(os.environ.get("RUN_CONFIG", Path(__file__).with_name("run_config.json"))).open() as f:
    CONFIG = json.load(f)

# Respect the node's cache configuration, with a configurable fallback.
os.environ.setdefault("TORCH_HOME", CONFIG["torch_home"])

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", CONFIG["data_dir"])
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", CONFIG["batch_size"]))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", CONFIG["num_workers"]))


class Tracking:
    """Tracking is optional; failures never interrupt prediction generation."""

    def __init__(self):
        self.client = None
        self.warned = False

    def disable(self, error):
        self.client = None
        if not self.warned:
            print(f"Warning: MLflow tracking disabled: {error}", flush=True)
            self.warned = True

    def call(self, method, *args, **kwargs):
        if self.client is not None:
            try:
                getattr(self.client, method)(*args, **kwargs)
            except Exception as error:
                self.disable(error)

    @contextmanager
    def run(self):
        context = None
        try:
            import mlflow
            context = mlflow.start_run()
            context.__enter__()
            self.client = mlflow
        except Exception as error:
            context = None
            self.disable(error)
        try:
            yield self
        finally:
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception as error:
                    self.disable(error)


@contextmanager
def worker_tempdir():
    # multiprocessing appends a socket suffix; the inherited run path can be too long.
    root = os.environ.get("WORKER_TMP_ROOT", CONFIG["worker_tmp_root"])
    with tempfile.TemporaryDirectory(prefix="dog-", dir=root) as path:
        old_env = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
        old_cached = tempfile.tempdir
        for key in old_env:
            os.environ[key] = path
        tempfile.tempdir = path
        try:
            yield
        finally:
            tempfile.tempdir = old_cached
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

TFM = transforms.Compose([
    transforms.Resize(219),
    transforms.CenterCrop(192),
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


@torch.no_grad()
def predict_two_view(model, clf, root, ids, device):
    loader = DataLoader(
        ImageFolderList(root, ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    probabilities = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            original = model(batch)
        p_original = clf.predict_proba(original.flatten(1).float().cpu().numpy())
        del original
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            flipped = model(batch.flip(-1))
        p_flipped = clf.predict_proba(flipped.flatten(1).float().cpu().numpy())
        del flipped
        probabilities.append(0.5 * p_original + 0.5 * p_flipped)
    return np.concatenate(probabilities, axis=0).astype(np.float64)


def main(tracking):
    started = time.monotonic()
    tracking.call("log_params", {
        "backbone": "resnet50", "weights": "IMAGENET1K_V1",
        "classifier": "LogisticRegression", "C": 0.3, "max_iter": 1000,
        "seed": SEED, "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
        "train_views": 1, "test_views": 2, "probability_average": "equal",
        "resize": 219, "center_crop": 192,
    })
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}", flush=True)
    for name, value in {"train_rows": len(labels), "test_rows": len(test_ids), "num_classes": len(breeds)}.items():
        tracking.call("log_metric", name, value, step=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()  # 2048-d pooled features
    net.eval().to(device)

    feature_started = time.monotonic()
    x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
    elapsed = time.monotonic() - feature_started
    print(f"training features ready in {elapsed:.1f}s", flush=True)
    tracking.call("log_metric", "train_feature_seconds", elapsed, step=0)

    # C picked once on a 15% stratified holdout over {0.03, 0.1, 0.3, 1, 3}; the
    # raw (un-normalised) 2048-d features need more L2 than the sklearn default.
    clf = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED)
    fit_started = time.monotonic()
    clf.fit(x_train, labels["breed"].values)
    elapsed = time.monotonic() - fit_started
    print(f"classifier fitted in {elapsed:.1f}s", flush=True)
    tracking.call("log_metric", "classifier_fit_seconds", elapsed, step=0)

    inference_started = time.monotonic()
    proba = predict_two_view(net, clf, os.path.join(PUBLIC, "test"), test_ids, device)
    elapsed = time.monotonic() - inference_started
    print(f"two-view inference completed in {elapsed:.1f}s", flush=True)
    tracking.call("log_metric", "test_inference_seconds", elapsed, step=0)
    # reorder model classes into the sample-submission column order
    order = [list(clf.classes_).index(b) for b in breeds]
    proba = proba[:, order]
    proba /= proba.sum(axis=1, keepdims=True)  # rows must sum to 1

    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # Graded only by the hill's evaluator; no in-run scoring.
    elapsed = time.monotonic() - started
    print(f"submission written in {elapsed:.1f}s", flush=True)
    tracking.call("log_metric", "total_seconds", elapsed, step=0)


if __name__ == "__main__":
    with worker_tempdir(), Tracking().run() as tracking:
        main(tracking)
