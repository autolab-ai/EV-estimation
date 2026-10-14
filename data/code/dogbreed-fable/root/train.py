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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = 8

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

    x_train = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
    x_test = extract(net, os.path.join(PUBLIC, "test"), test_ids, device)
    print(f"features: train={x_train.shape} test={x_test.shape}")

    # C picked once on a 15% stratified holdout over {0.03, 0.1, 0.3, 1, 3}; the
    # raw (un-normalised) 2048-d features need more L2 than the sklearn default.
    clf = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED)
    clf.fit(x_train, labels["breed"].values)

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
