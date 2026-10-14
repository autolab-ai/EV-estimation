"""Baseline for dog-breed-identification.

Frozen-backbone transfer learning: a timm ConvNeXt-Base (fb_in22k_ft_in1k)
is used purely as a feature extractor (no fine-tuning). Every train and test
image is resized/center-cropped to 224px, pushed through the network, and the
1024-d pooled pre-logit feature is kept. A plain sklearn LogisticRegression,
with C picked on a 15% stratified holdout, is then fitted on those raw features
to predict the 120 breeds.

Deliberately plain: no augmentation, no test-time augmentation, no feature
normalisation, no multi-backbone concatenation, no fine-tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

# Must be set before torchvision is imported: pretrained weights are cached here
# instead of in the (quota-limited) login-node home directory.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")
# timm/HF hub checkpoints (~350 MB) go to scratch too.
os.environ.setdefault("HF_HOME", "<HF_HOME>")
# The evaluator sometimes launches with a very long TMPDIR; multiprocessing's
# Unix listener socket under it exceeds the 108-char path limit, DataLoader
# workers die with "AF_UNIX path too long" and the run hangs. Use a short dir.
import tempfile

if len(tempfile.gettempdir()) > 40:
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"

import re
import subprocess

import numpy as np
import pandas as pd
import torch
import timm
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class _Tracker:
    """Best-effort MLflow logging: disables itself after the first failure."""

    def __init__(self):
        self.ok = mlflow is not None
        self.run = None

    def start(self):
        if not self.ok:
            return
        try:
            self.run = mlflow.start_run()
        except Exception as e:
            print(f"[mlflow] disabled: {e}")
            self.ok = False

    def params(self, d):
        if not self.ok:
            return
        try:
            mlflow.log_params(d)
        except Exception as e:
            print(f"[mlflow] disabled: {e}")
            self.ok = False

    def metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print(f"[mlflow] disabled: {e}")
            self.ok = False

    def end(self):
        if self.ok and self.run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
BACKBONE = "convnext_base.fb_in22k_ft_in1k"
C_GRID = [0.03, 0.1, 0.3, 1.0]
HOLDOUT = 0.15

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
        # Python 3.14 defaults to forkserver; fork keeps worker startup cheap
        # and version-independent (workers never touch CUDA).
        multiprocessing_context="fork" if NUM_WORKERS > 0 else None,
    )
    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        feats.append(out.flatten(1).float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tracker = _Tracker()
    tracker.start()
    tracker.params({"backbone": BACKBONE, "batch_size": BATCH_SIZE, "holdout": HOLDOUT,
                    "c_grid": str(C_GRID), "img_size": 224})

    # num_classes=0 -> 1024-d globally pooled, layer-normed pre-logit features.
    net = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    net.eval().to(device)

    x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
    x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device)
    print(f"features: train={x_train.shape} test={x_test.shape}")

    # Plain numpy labels: pandas>=3 returns an Arrow-backed string array from
    # .values, which sklearn's stratified split cannot fancy-index.
    y_train = labels["breed"].to_numpy(dtype=object)

    # Re-select C for the new feature scale on a stratified 15% holdout.
    x_tr, x_ho, y_tr, y_ho = train_test_split(
        x_train, y_train, test_size=HOLDOUT, stratify=y_train, random_state=SEED)
    best_c, best_ll = None, float("inf")
    for i, c in enumerate(C_GRID):
        m = LogisticRegression(C=c, max_iter=1000, random_state=SEED).fit(x_tr, y_tr)
        ll = log_loss(y_ho, m.predict_proba(x_ho), labels=m.classes_)
        print(f"holdout C={c}: log_loss={ll:.4f}")
        tracker.metric("holdout_log_loss", ll, step=i)
        if ll < best_ll:
            best_c, best_ll = c, ll
    print(f"best C={best_c} holdout log_loss={best_ll:.4f}")
    tracker.metric("best_holdout_log_loss", best_ll)
    tracker.params({"best_C": best_c})

    clf = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
    clf.fit(x_train, y_train)

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
    tracker.end()


if __name__ == "__main__":
    main()
