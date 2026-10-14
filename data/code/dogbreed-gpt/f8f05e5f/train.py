"""Baseline for dog-breed-identification.

Frozen-backbone transfer learning: a torchvision ResNet-50 with IMAGENET1K_V2
weights is used purely as a feature extractor (no fine-tuning). Every train and
test image is resized/center-cropped to 224px, pushed through the network, and
the 2048-d global-average-pooled activation is kept. A 256-dimensional PCA
projection is fitted on training features only. A plain sklearn
LogisticRegression (C=0.3) is then fitted on the projected features to predict
the 120 breeds.

Deliberately plain: no augmentation, no test-time augmentation, no feature
normalisation, no multi-backbone concatenation, no fine-tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tracking import Tracking

with open(os.environ.get("TRAIN_CONFIG", Path(__file__).with_name("train_config.json"))) as f:
    CONFIG = json.load(f)

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", CONFIG["data_dir"])
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", CONFIG["batch_size"]))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", CONFIG["num_workers"]))

WEIGHTS = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
TFM = WEIGHTS.transforms()


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


def main(tracking):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tracking.log_params({
        "model": "resnet50", "weights": WEIGHTS.name,
        "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
        "resize": 232, "crop": 224, "seed": SEED,
        "feature_dim": 2048, "classifier_C": 0.3,
        "classifier_solver": "lbfgs", "classifier_max_iter": 1000,
        "pca_components": 256, "pca_solver": "randomized",
        "pca_whiten": False, "pca_iterated_power": 3, "pca_random_state": 0,
    })

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}", flush=True)
    tracking.log_metric("train_images", len(labels))
    tracking.log_metric("test_images", len(test_ids))
    tracking.log_metric("num_classes", len(breeds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    net = torchvision.models.resnet50(weights=WEIGHTS)
    net.fc = torch.nn.Identity()  # 2048-d pooled features
    net.eval().to(device)

    started = time.monotonic()
    x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
    tracking.log_metric("train_extraction_seconds", time.monotonic() - started)
    started = time.monotonic()
    x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device)
    tracking.log_metric("test_extraction_seconds", time.monotonic() - started)
    print(f"features: train={x_train.shape} test={x_test.shape}", flush=True)
    tracking.log_metric("feature_dim", x_train.shape[1])

    pca = PCA(
        n_components=256, svd_solver="randomized", whiten=False,
        iterated_power=3, random_state=0,
    )
    started = time.monotonic()
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    tracking.log_metric("pca_fit_transform_seconds", time.monotonic() - started)
    tracking.log_metric("projected_feature_dim", x_train.shape[1])
    print(f"projected features: train={x_train.shape} test={x_test.shape}", flush=True)

    clf = LogisticRegression(C=0.3, solver="lbfgs", max_iter=1000, random_state=SEED)
    started = time.monotonic()
    clf.fit(x_train, labels["breed"].values)
    tracking.log_metric("classifier_fit_seconds", time.monotonic() - started)
    tracking.log_metric("classifier_iterations", int(clf.n_iter_.max()))

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

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    # Multiprocessing sockets must not inherit a worktree-length temp path.
    with tempfile.TemporaryDirectory(
        prefix="dog-", dir=os.environ.get("WORKER_TEMP_ROOT", CONFIG["temp_root"]),
    ) as temp_dir:
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = temp_dir
        tempfile.tempdir = temp_dir
        with Tracking() as tracking:
            main(tracking)
