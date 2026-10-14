"""Two-backbone frozen-feature baseline for dog-breed-identification (EV Bench L1 node).

Frozen-backbone transfer learning with two architecturally distinct extractors:

1. torchvision ResNet-50 (IMAGENET1K_V1) -> 2048-d global-average-pooled features
   (extraction identical to the root baseline: Resize 256 / CenterCrop 224).
2. timm ConvNeXt-Tiny (convnext_tiny.fb_in22k_ft_in1k, num_classes=0) -> 768-d
   pooled features, with its own eval transform resolved from the model's
   pretrained config.

Each feature block is row-wise L2-normalised independently (train and test
separately), then the blocks are concatenated horizontally into a 2816-d
feature per image. A plain sklearn LogisticRegression (C=1.0) is fitted on the
concatenated train features to predict the 120 breeds.

No fine-tuning, no test-time augmentation.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

# Must be set before torchvision is imported: pretrained weights are cached here
# instead of in the (quota-limited) login-node home directory.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")

import contextlib
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm

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


# --- MLflow: best-effort telemetry; must never break the run -----------------
class _MLflow:
    def __init__(self):
        self.enabled = False
        self.ctx = contextlib.nullcontext()
        try:
            import mlflow  # noqa: F401
            self._mlflow = mlflow
            self.ctx = mlflow.start_run()
            self.enabled = True
        except Exception as exc:  # unreachable server, expired auth, ...
            print(f"mlflow disabled: {exc}")

    def log_params(self, params):
        if self.enabled:
            try:
                self._mlflow.log_params(params)
            except Exception as exc:
                print(f"mlflow log_params failed: {exc}")
                self.enabled = False

    def log_metric(self, name, value, step=None):
        if self.enabled:
            try:
                self._mlflow.log_metric(name, value, step=step)
            except Exception as exc:
                print(f"mlflow log_metric failed: {exc}")
                self.enabled = False


class ImageFolderList(Dataset):
    """Loads <root>/<id>.jpg for a list of ids, in the given order."""

    def __init__(self, root, ids, tfm=TFM):
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
def extract(model, root, ids, device, tfm=TFM):
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

    mlf = _MLflow()
    with mlf.ctx:
        mlf.log_params({
            "model": "resnet50_imagenet1k_v1+convnext_tiny.fb_in22k_ft_in1k",
            "feature_dim": 2816,
            "C": 1.0,
            "max_iter": 1000,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "feature_norm": "l2_per_block",
        })

        labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
        sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
        breeds = [c for c in sample.columns if c != "id"]
        test_ids = sample["id"].tolist()
        print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device}")

        t0 = time.time()
        # Backbone 1: ResNet-50, 2048-d (identical to the root baseline)
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet50(weights=weights)
        net.fc = torch.nn.Identity()  # 2048-d pooled features
        net.eval().to(device)
        x_train_r = extract(net, os.path.join(PUBLIC, "train"), labels["id"], device)
        x_test_r = extract(net, os.path.join(PUBLIC, "test"), test_ids, device)

        # Backbone 2: ConvNeXt-Tiny (in22k fine-tuned on in1k), 768-d
        model2 = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0)
        model2.eval().to(device)
        cfg = timm.data.resolve_data_config(model2.pretrained_cfg)
        tfm2 = timm.data.create_transform(**cfg, is_training=False)
        x_train_c = extract(model2, os.path.join(PUBLIC, "train"), labels["id"], device, tfm=tfm2)
        x_test_c = extract(model2, os.path.join(PUBLIC, "test"), test_ids, device, tfm=tfm2)
        extract_s = time.time() - t0
        print(f"features: resnet train={x_train_r.shape} convnext train={x_train_c.shape} "
              f"({extract_s:.0f}s)")
        mlf.log_metric("feature_extract_seconds", extract_s)

        # Row-wise L2-normalise each block independently, then concatenate.
        x_train = np.concatenate(
            [normalize(x_train_r, axis=1), normalize(x_train_c, axis=1)], axis=1)
        x_test = np.concatenate(
            [normalize(x_test_r, axis=1), normalize(x_test_c, axis=1)], axis=1)
        print(f"concatenated: train={x_train.shape} test={x_test.shape}")
        del x_train_r, x_test_r, x_train_c, x_test_c

        t0 = time.time()
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        clf.fit(x_train, labels["breed"].values)
        fit_s = time.time() - t0
        print(f"logreg fit: {fit_s:.0f}s, n_iter={clf.n_iter_}")
        mlf.log_metric("fit_seconds", fit_s)
        mlf.log_metric("fit_n_iter", int(clf.n_iter_[0]))

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
