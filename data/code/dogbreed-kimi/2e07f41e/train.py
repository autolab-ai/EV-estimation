"""Frozen-backbone transfer learning for dog-breed-identification.

Backbone swap experiment: a timm ViT-B/16 (vit_base_patch16_224.augreg_in21k_ft_in1k,
ImageNet-21k pretrain fine-tuned on ImageNet-1k) is used purely as a feature
extractor (no fine-tuning). Every train and test image is preprocessed with the
transform resolved from the model's own pretrained config (so normalization
matches pretraining), pushed through the network, and the 768-d pooled feature
(num_classes=0) is kept. A plain sklearn LogisticRegression (C=0.3) is then
fitted on those raw features to predict the 120 breeds.

Deliberately plain apart from the backbone: no augmentation, no test-time
augmentation, no feature normalisation, no fine-tuning.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

# Must be set before any torch/hub imports: pretrained weights are cached here
# instead of in the (quota-limited) login-node home directory.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")

import contextlib
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = 8
BACKBONE = "vit_base_patch16_224.augreg_in21k_ft_in1k"
LR_C = 0.3
LR_MAX_ITER = 1000


# --- MLflow: best-effort, never allowed to kill the run ---------------------
try:
    import mlflow
    _mlflow_ok = True
except Exception as _e:  # pragma: no cover
    print(f"mlflow unavailable, running untracked: {_e}")
    mlflow = None
    _mlflow_ok = False


def mlflow_run():
    """start_run() if possible, else a no-op context."""
    if not _mlflow_ok:
        return contextlib.nullcontext()
    try:
        return mlflow.start_run()
    except Exception as e:
        print(f"mlflow start_run failed, running untracked: {e}")
        return contextlib.nullcontext()


def ml_log_params(params):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        mlflow.log_params(params)
    except Exception as e:
        _mlflow_ok = False
        print(f"mlflow logging disabled after error: {e}")


def ml_log_metric(name, value, step=None):
    global _mlflow_ok
    if not _mlflow_ok:
        return
    try:
        mlflow.log_metric(name, value, step=step)
    except Exception as e:
        _mlflow_ok = False
        print(f"mlflow logging disabled after error: {e}")


class ImageFolderList(Dataset):
    """Loads <root>/<id>.jpg for a list of ids, in the given order."""

    def __init__(self, root, ids, tfm):
        self.root = root
        self.ids = list(ids)
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            return self.tfm(img.convert("RGB"))


@torch.no_grad()
def extract(model, root, ids, device, tfm):
    loader = DataLoader(
        ImageFolderList(root, ids, tfm), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
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

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval().to(device)  # num_classes=0 -> 768-d pooled features directly

    # Preprocessing resolved from the model's own config so normalization
    # matches pretraining.
    cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    tfm = timm.data.create_transform(**cfg, is_training=False)

    with mlflow_run():
        ml_log_params({
            "backbone": BACKBONE,
            "lr_C": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "input_size": tuple(cfg["input_size"]),
        })

        t0 = time.time()
        x_train = extract(model, os.path.join(PUBLIC, "train"), labels["id"], device, tfm)
        x_test = extract(model, os.path.join(PUBLIC, "test"), test_ids, device, tfm)
        extract_s = time.time() - t0
        print(f"features: train={x_train.shape} test={x_test.shape} ({extract_s:.1f}s)")
        ml_log_metric("feature_dim", x_train.shape[1])
        ml_log_metric("feature_extract_seconds", extract_s)

        # C picked once on a 15% stratified holdout over {0.03, 0.1, 0.3, 1, 3};
        # the raw (un-normalised) pooled features need more L2 than the sklearn
        # default.
        t0 = time.time()
        clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, random_state=SEED)
        clf.fit(x_train, labels["breed"].values)
        ml_log_metric("logreg_fit_seconds", time.time() - t0)
        ml_log_metric("logreg_n_iter", float(clf.n_iter_[0]))

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
    main()
