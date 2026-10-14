"""Baseline for dog-breed-identification.

Frozen-backbone transfer learning: a torchvision ResNet-50 with IMAGENET1K_V1
weights is used purely as a feature extractor (no fine-tuning). Every train and
test image is resized/center-cropped to 224px, pushed through the network, and
the 2048-d global-average-pooled activation is kept. A plain sklearn
LogisticRegression (C=0.3) is then fitted on those raw features to predict the
120 breeds.

Deliberately plain: no augmentation, no test-time augmentation, no feature
normalisation, no multi-backbone concatenation, no fine-tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import tempfile

# The node's default TMPDIR can exceed the 108-byte AF_UNIX socket path limit,
# which silently kills DataLoader workers; use a short temp dir instead.
os.environ["TMPDIR"] = os.environ.get("EV_TMPDIR", "/tmp")
tempfile.tempdir = None

# Must be set before torchvision is imported: pretrained weights are cached here
# instead of in the (quota-limited) login-node home directory.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")

import re
import subprocess

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = int(os.environ.get("EV_NUM_WORKERS", "8"))
C_GRID = [0.003, 0.01, 0.03, 0.1]
HOLDOUT_FRAC = 0.15


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
            print(f"[mlflow] disabled: {e}", flush=True)
            self.ok = False

    def params(self, d):
        if not self.ok:
            return
        try:
            mlflow.log_params(d)
        except Exception as e:
            print(f"[mlflow] disabled: {e}", flush=True)
            self.ok = False

    def metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print(f"[mlflow] disabled: {e}", flush=True)
            self.ok = False

    def end(self):
        if self.ok and self.run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


TRACK = _Tracker()

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


def resnet_forward(net, x):
    """torchvision ResNet forward returning (2048-d pooled, 1000-d logits)."""
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    pooled = torch.flatten(net.avgpool(x), 1)
    logits = net.fc(pooled)
    return pooled, logits


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
            pooled, _logits = resnet_forward(model, batch)
        # linear probe on the penultimate representation only: the 1000-d
        # ImageNet log-proba block is a deterministic function of `pooled`
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}", flush=True)
    TRACK.start()
    TRACK.params({"backbone": "resnet50_IMAGENET1K_V1", "features": "pooled2048",
                  "scaler": "standard", "c_grid": str(C_GRID), "holdout_frac": HOLDOUT_FRAC,
                  "batch_size": BATCH_SIZE, "seed": SEED})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=weights)
    # net.fc is kept (helper computes logits) but only the pooled block is used
    net.eval().to(device)

    x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
    x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device)
    print(f"features: train={x_train.shape} test={x_test.shape}", flush=True)

    # standardise the pooled features (fit on train only)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y = np.asarray(labels["breed"].astype(str).tolist(), dtype=object)

    # select C on a stratified holdout by multiclass log-loss
    xtr, xho, ytr, yho = train_test_split(
        x_train, y, test_size=HOLDOUT_FRAC, stratify=y, random_state=SEED)
    best_c, best_ll = None, float("inf")
    for i, c in enumerate(C_GRID):
        m = LogisticRegression(C=c, max_iter=1000, random_state=SEED).fit(xtr, ytr)
        ll = log_loss(yho, m.predict_proba(xho), labels=m.classes_)
        print(f"holdout C={c}: log_loss={ll:.4f}", flush=True)
        TRACK.metric("holdout_log_loss", ll, step=i)
        if ll < best_ll:
            best_c, best_ll = c, ll
    print(f"best C={best_c} holdout log_loss={best_ll:.4f}", flush=True)
    TRACK.metric("best_holdout_log_loss", best_ll)
    TRACK.params({"best_C": best_c})

    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
    clf.fit(x_train, y)

    proba = clf.predict_proba(x_test).astype(np.float64)
    # reorder model classes into the sample-submission column order
    order = [list(clf.classes_).index(b) for b in breeds]
    proba = proba[:, order]
    proba /= proba.sum(axis=1, keepdims=True)  # rows must sum to 1

    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    print("wrote submission/submission.csv", flush=True)
    TRACK.end()
    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
