"""Frozen DINOv2 ViT-B/14 feature extractor + logistic regression (EV Bench node L1-a99c-r3.1).

Same recipe as the frozen-backbone baseline, with ONE change: the torchvision
ResNet-50 (IMAGENET1K_V1) feature extractor is replaced by a frozen,
self-supervised DINOv2 ViT-B/14 (timm "vit_base_patch14_dinov2.lvd142m"),
which produces much stronger embeddings for fine-grained categories such as
dog breeds. Everything else is structurally identical:

- every train image is resized to 256 and center-cropped to 224px, pushed
  through the frozen network, and the pooled embedding (768-d instead of
  2048-d) is kept;
- a plain sklearn LogisticRegression is fitted on those raw single-view
  features, with C re-selected on a single stratified 15% holdout of the
  TRAIN features over {0.03, 0.1, 0.3, 1, 3} (the optimal L2 strength differs
  for the new features), then refit on all training features;
- test-time inference uses the same 10-view TTA (Resize 256 -> FiveCrop 224,
  each crop + its hflip), clf.predict_proba averaged over the 10 views.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

# The harness exports TMPDIR as a path longer than the AF_UNIX sun_path limit,
# which silently hangs PyTorch DataLoader workers (their resource_sharer socket
# cannot be created). setdefault is a no-op because the variable always exists,
# so this MUST be a hard assignment, and it must happen before torch import.
os.environ["TMPDIR"] = "/tmp"

# Pretrained weights are cached here instead of in the (quota-limited)
# login-node home directory; harmless if the environment already pins one.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")

import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = 8

# Normalisation constants come from the timm model config in main() and are
# passed down explicitly: DataLoader workers are spawned via forkserver on
# Python 3.14 (module globals mutated in main() would be None there), so the
# transforms live on the pickled Dataset instance instead of in globals.
_TO_TENSOR = transforms.ToTensor()
_RESIZE_256 = transforms.Resize(256)
_FIVE_CROP = transforms.FiveCrop(224)


def ten_view_tta(img, normalize):
    """Test-only transform: Resize(256) -> FiveCrop(224) -> each crop + its
    horizontal flip -> ToTensor + norm, stacked to (10, 3, 224, 224)."""
    img = _RESIZE_256(img)
    views = []
    for crop in _FIVE_CROP(img):
        for view in (crop, TF.hflip(crop)):
            views.append(normalize(_TO_TENSOR(view)))
    return torch.stack(views)


class ImageFolderList(Dataset):
    """Loads <root>/<id>.jpg for a list of ids, in the given order.

    tta=False (train path, unchanged): returns a single (3, 224, 224) tensor.
    tta=True (test path): returns the stacked (10, 3, 224, 224) TTA tensor.
    """

    def __init__(self, root, ids, tta=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.root = root
        self.ids = list(ids)
        self.tta = tta
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            img = img.convert("RGB")
            return ten_view_tta(img, self.normalize) if self.tta else self.tfm(img)


@torch.no_grad()
def extract(model, root, ids, device, tta=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    loader = DataLoader(
        ImageFolderList(root, ids, tta=tta, mean=mean, std=std),
        batch_size=BATCH_SIZE, shuffle=False,
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
            out = out.reshape(b, v, -1)  # back to (B, 10, feat_dim)
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

    # frozen DINOv2 ViT-B/14 at 224px (timm interpolates the position
    # embeddings; the model's native default is 518). num_classes=0 returns
    # the 768-d pooled embedding.
    net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                            num_classes=0, img_size=224)
    net.eval().to(device)
    cfg = timm.data.resolve_data_config(model=net)

    with mlflow_start():
        mlflow_log("log_params", {
            "backbone": "vit_base_patch14_dinov2.lvd142m-frozen",
            "classifier": "LogisticRegression",
            "C_grid": "0.03,0.1,0.3,1,3",
            "tta_views": 10,
            "tta_recipe": "resize256+fivecrop224+hflip",
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        })

        t0 = time.time()
        x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device,
                          mean=cfg["mean"], std=cfg["std"])
        t1 = time.time()
        x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device, tta=True,
                         mean=cfg["mean"], std=cfg["std"])
        t2 = time.time()
        print(f"features: train={x_train.shape} test={x_test.shape}")
        mlflow_log("log_metric", "train_extract_seconds", t1 - t0)
        mlflow_log("log_metric", "test_extract_seconds", t2 - t1)

        # plain numpy object array: pandas 3.x returns Arrow-backed arrays from
        # .values on string columns, which sklearn cannot fancy-index.
        y = labels["breed"].astype(object).to_numpy()

        # Re-select C once on a single stratified 15% holdout of the TRAIN
        # features (the optimal L2 strength differs for the 768-d DINOv2
        # features), then refit on all training features with the chosen C.
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y, test_size=0.15, random_state=SEED, stratify=y)
        best_c, best_ll = None, None
        for c in (0.03, 0.1, 0.3, 1, 3):
            probe = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
            probe.fit(x_tr, y_tr)
            ll = log_loss(y_val, probe.predict_proba(x_val), labels=probe.classes_)
            print(f"C={c}: holdout_log_loss={ll:.4f}", flush=True)
            mlflow_log("log_metric", "holdout_log_loss", ll, step=int(c * 1000))
            if best_ll is None or ll < best_ll:
                best_c, best_ll = c, ll
        print(f"selected C={best_c} (holdout log-loss {best_ll:.4f})", flush=True)
        mlflow_log("log_params", {"C": best_c})
        mlflow_log("log_metric", "best_holdout_log_loss", best_ll)

        clf = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
        clf.fit(x_train, y)
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
