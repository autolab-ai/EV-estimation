"""10-view test-time augmentation on the frozen-backbone baseline (EV Bench node L1-1861-1).

Training path is IDENTICAL to the baseline: a torchvision ResNet-50 with
IMAGENET1K_V1 weights is used purely as a feature extractor (no fine-tuning);
every train image is resized to 256 and center-cropped to 224px, pushed through
the network, and the 2048-d global-average-pooled activation is kept. A plain
sklearn LogisticRegression (C=0.3) is fitted on those raw single-view features.

Only the test-time inference path changes: each test image is resized to 256,
then FiveCrop(224) yields 5 crops (4 corners + center), and each crop is also
horizontally flipped, giving 10 views per image. Each view goes through the
same ToTensor + ImageNet normalisation as training, all 10 views are forwarded
through the frozen backbone, and clf.predict_proba is averaged (arithmetic
mean) over the 10 views.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

# Must be set before torchvision is imported: pretrained weights are cached here
# instead of in the (quota-limited) login-node home directory.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")

import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_TO_TENSOR = transforms.ToTensor()
_RESIZE_256 = transforms.Resize(256)
_FIVE_CROP = transforms.FiveCrop(224)


def ten_view_tta(img):
    """Test-only transform: Resize(256) -> FiveCrop(224) -> each crop + its
    horizontal flip -> ToTensor + ImageNet norm, stacked to (10, 3, 224, 224)."""
    img = _RESIZE_256(img)
    views = []
    for crop in _FIVE_CROP(img):
        for view in (crop, TF.hflip(crop)):
            views.append(_NORMALIZE(_TO_TENSOR(view)))
    return torch.stack(views)


class ImageFolderList(Dataset):
    """Loads <root>/<id>.jpg for a list of ids, in the given order.

    tta=False (train path, unchanged): returns a single (3, 224, 224) tensor.
    tta=True (test path): returns the stacked (10, 3, 224, 224) TTA tensor.
    """

    def __init__(self, root, ids, tta=False):
        self.root = root
        self.ids = list(ids)
        self.tta = tta

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            img = img.convert("RGB")
            return ten_view_tta(img) if self.tta else TFM(img)


@torch.no_grad()
def extract(model, root, ids, device, tta=False):
    loader = DataLoader(
        ImageFolderList(root, ids, tta=tta), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if tta:
            # (B, 10, 3, 224, 224) -> (B*10, 3, 224, 224) for the forward pass
            b, v = batch.shape[0], batch.shape[1]
            batch = batch.reshape(b * v, *batch.shape[2:])
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        out = out.flatten(1).float()
        if tta:
            out = out.reshape(b, v, -1)  # back to (B, 10, 2048)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


# --- MLflow telemetry: best-effort only, must never kill the run ------------
try:
    import mlflow  # noqa: F401
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False


class _NoOpRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def mlflow_start():
    if not _MLFLOW_OK:
        return _NoOpRun()
    try:
        return mlflow.start_run()
    except Exception as e:
        print(f"mlflow disabled: {e}")
        return _NoOpRun()


def mlflow_log(fn, *args, **kwargs):
    global _MLFLOW_OK
    if not _MLFLOW_OK:
        return
    try:
        getattr(mlflow, fn)(*args, **kwargs)
    except Exception as e:
        print(f"mlflow disabled: {e}")
        _MLFLOW_OK = False


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

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()  # 2048-d pooled features
    net.eval().to(device)

    with mlflow_start():
        mlflow_log("log_params", {
            "backbone": "resnet50-IMAGENET1K_V1-frozen",
            "classifier": "LogisticRegression",
            "C": 0.3,
            "tta_views": 10,
            "tta_recipe": "resize256+fivecrop224+hflip",
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        })

        t0 = time.time()
        x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
        t1 = time.time()
        x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device, tta=True)
        t2 = time.time()
        print(f"features: train={x_train.shape} test={x_test.shape}")
        mlflow_log("log_metric", "train_extract_seconds", t1 - t0)
        mlflow_log("log_metric", "test_extract_seconds", t2 - t1)

        # C picked once on a 15% stratified holdout over {0.03, 0.1, 0.3, 1, 3}; the
        # raw (un-normalised) 2048-d features need more L2 than the sklearn default.
        clf = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED)
        clf.fit(x_train, labels["breed"].values)
        mlflow_log("log_metric", "fit_seconds", time.time() - t2)

        # predict per-view probabilities and average over the 10 TTA views
        n_test = len(test_ids)
        proba = clf.predict_proba(x_test.reshape(n_test * 10, -1)).astype(np.float64)
        proba = proba.reshape(n_test, 10, -1).mean(axis=1)
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
    main()
