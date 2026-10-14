"""dog-breed-identification: frozen features from three diverse timm backbones.

Replaces the single ResNet-50 feature vector with the concatenation of frozen
features from three backbones pretrained with different signals:
  (1) convnext_base.fb_in22k_ft_in1k   supervised ImageNet-22k -> 1024-d pooled
  (2) vit_base_patch16_clip_224.openai  image-text contrastive  ->  768-d CLS
  (3) vit_base_patch14_dinov2.lvd142m   self-supervised         ->  768-d CLS
Each block is L2-normalised row-wise and the blocks are concatenated (2560-d).
A LogisticRegression is fitted on top; C is picked on a 15% stratified holdout
and the classifier is then refitted on all training rows.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import tempfile

# TMPDIR may be pinned to a path longer than the 108-char AF_UNIX limit, which
# silently kills DataLoader workers. Use /tmp for multiprocessing sockets.
os.environ["TMPDIR"] = "/tmp"
tempfile.tempdir = None
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")
os.environ.setdefault("HF_HOME", "<HF_HOME>")

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
C_GRID = [1.0, 3.0, 10.0, 30.0]

BASE_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
CLIP_TFM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758)),
])


class MLFlowLogger:
    """Best-effort MLflow wrapper: disables itself after the first failure."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self.run = None
        try:
            import mlflow
            self.mlflow = mlflow
            self.run = mlflow.start_run()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] disabled: {e}")

    def log_params(self, d):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] disabled after error: {e}")
            self.ok = False

    def log_metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] disabled after error: {e}")
            self.ok = False

    def end(self):
        if self.ok:
            try:
                self.mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


class ImageFolderList(Dataset):
    """Loads <root>/<id>.jpg once and returns (baseline_tensor, clip_tensor)."""

    def __init__(self, root, ids):
        self.root = root
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            return BASE_TFM(rgb), CLIP_TFM(rgb)


def build_models(device):
    convnext = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=0)
    clip = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0)
    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    models = [m.eval().to(device) for m in (convnext, clip, dino)]
    return models


@torch.no_grad()
def extract(models, root, ids, device):
    loader = DataLoader(
        ImageFolderList(root, ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        timeout=120 if NUM_WORKERS > 0 else 0,
    )
    convnext, clip, dino = models
    feats = [[], [], []]
    for base, clipx in loader:
        base = base.to(device, non_blocking=True)
        clipx = clipx.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f1 = convnext(base)
            f2 = clip(clipx)
            f3 = dino(base)
        for store, f in zip(feats, (f1, f2, f3)):
            store.append(f.flatten(1).float().cpu().numpy())
    blocks = [normalize(np.concatenate(s, axis=0)) for s in feats]
    return np.concatenate(blocks, axis=1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ml = MLFlowLogger()
    ml.log_params({
        "backbones": "convnext_base.fb_in22k_ft_in1k+vit_base_patch16_clip_224.openai+vit_base_patch14_dinov2.lvd142m",
        "batch_size": BATCH_SIZE, "c_grid": str(C_GRID), "holdout": 0.15, "seed": SEED,
    })

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    # plain numpy object array: pandas>=3 pyarrow-backed strings break sklearn indexing
    y = np.asarray(labels["breed"].tolist(), dtype=object)
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    models = build_models(device)
    x_train = extract(models, os.path.join(PUBLIC, "train"), labels["id"], device)
    x_test = extract(models, os.path.join(PUBLIC, "test"), test_ids, device)
    print(f"features: train={x_train.shape} test={x_test.shape}", flush=True)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    xtr, xho, ytr, yho = train_test_split(
        x_train, y, test_size=0.15, stratify=y, random_state=SEED)
    best_c, best_ll = None, float("inf")
    for step, c in enumerate(C_GRID):
        clf = LogisticRegression(C=c, max_iter=500, random_state=SEED)
        clf.fit(xtr, ytr)
        ll = log_loss(yho, clf.predict_proba(xho), labels=clf.classes_)
        print(f"C={c}: holdout log-loss={ll:.4f}", flush=True)
        ml.log_metric("holdout_log_loss", ll, step=step)
        if ll < best_ll:
            best_c, best_ll = c, ll
    print(f"best C={best_c} holdout log-loss={best_ll:.4f}", flush=True)
    ml.log_metric("best_holdout_log_loss", best_ll)
    ml.log_metric("best_C", best_c)

    clf = LogisticRegression(C=best_c, max_iter=500, random_state=SEED)
    clf.fit(x_train, y)

    proba = clf.predict_proba(x_test).astype(np.float64)
    order = [list(clf.classes_).index(b) for b in breeds]
    proba = proba[:, order]
    proba /= proba.sum(axis=1, keepdims=True)

    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    ml.end()


if __name__ == "__main__":
    main()
