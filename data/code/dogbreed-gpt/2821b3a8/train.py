"""Blend a learned logistic classifier with pretrained ResNet-50 dog probabilities.

The two classifiers share the baseline's frozen 2048-dimensional image features.
One stratified training-data split selects the blend weight and temperature;
the learned classifier is then refitted on all labeled images.

Writes submission.csv and submission/submission.csv; only the evaluator grades
held-out predictions. No held-out labels are read.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits
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
CPU_THREADS = int(os.environ.get("CPU_THREADS", CONFIG["cpu_threads"]))

WEIGHTS = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
TFM = WEIGHTS.transforms()
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TEMPERATURES = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0]


def dog_class_indices(breeds):
    categories = WEIGHTS.meta["categories"]
    candidates = list(range(151, 269)) + [273, 274, 275]
    indices = []
    for breed in breeds:
        matches = [i for i in candidates
                   if categories[i].lower().replace(" ", "_") == breed]
        assert len(matches) == 1, f"Expected one ImageNet dog match for {breed}: {matches}"
        indices.append(matches[0])
    assert len(indices) == len(set(indices)) == 120
    return indices


def dog_probabilities(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def new_classifier():
    return LogisticRegression(C=0.3, solver="lbfgs", max_iter=1000, random_state=SEED)


def ordered_probabilities(classifier, features, breeds):
    order = [list(classifier.classes_).index(breed) for breed in breeds]
    return classifier.predict_proba(features).astype(np.float64)[:, order]


def select_blend(lr_probabilities, dog_logits, correct_columns):
    """Choose calibration parameters on internal training validation rows only."""
    rows = np.arange(len(correct_columns))
    dog_distributions = {t: dog_probabilities(dog_logits, t) for t in TEMPERATURES}
    best_loss = float("inf")
    best_pair = None
    for alpha in ALPHAS:
        for temperature in TEMPERATURES:
            blended = (1.0 - alpha) * lr_probabilities + alpha * dog_distributions[temperature]
            loss = -np.log(np.clip(blended[rows, correct_columns], 1e-15, None)).mean()
            if loss < best_loss:
                best_loss = loss
                best_pair = (alpha, temperature)
    assert best_pair is not None
    return best_pair


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
        "cpu_threads": CPU_THREADS, "validation_fraction": 0.2,
        "blend_alphas": ALPHAS, "blend_temperatures": TEMPERATURES,
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

    selected_indices = dog_class_indices(breeds)
    net = torchvision.models.resnet50(weights=WEIGHTS)
    pretrained_weights = net.fc.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    pretrained_bias = net.fc.bias.detach().cpu().numpy().astype(np.float32, copy=True)
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

    dog_weights = pretrained_weights[selected_indices]
    dog_bias = pretrained_bias[selected_indices]
    z_train = x_train @ dog_weights.T + dog_bias
    z_test = x_test @ dog_weights.T + dog_bias

    targets = labels["breed"].values
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit_rows, validation_rows = next(split.split(x_train, targets))
    calibration_clf = new_classifier()
    started = time.monotonic()
    calibration_clf.fit(x_train[fit_rows], targets[fit_rows])
    tracking.log_metric("calibration_fit_seconds", time.monotonic() - started)
    tracking.log_metric("calibration_iterations", int(calibration_clf.n_iter_.max()))
    p_lr = ordered_probabilities(calibration_clf, x_train[validation_rows], breeds)
    breed_columns = {breed: i for i, breed in enumerate(breeds)}
    correct_columns = np.array([breed_columns[breed] for breed in targets[validation_rows]])
    alpha, temperature = select_blend(p_lr, z_train[validation_rows], correct_columns)
    print(f"Selected blend: pretrained_weight={alpha} temperature={temperature}", flush=True)
    tracking.log_metric("blend_alpha", alpha)
    tracking.log_metric("blend_temperature", temperature)
    del calibration_clf

    clf = new_classifier()
    started = time.monotonic()
    clf.fit(x_train, targets)
    tracking.log_metric("classifier_fit_seconds", time.monotonic() - started)
    tracking.log_metric("classifier_iterations", int(clf.n_iter_.max()))

    proba = ((1.0 - alpha) * ordered_probabilities(clf, x_test, breeds)
             + alpha * dog_probabilities(z_test, temperature))
    proba /= proba.sum(axis=1, keepdims=True)  # rows must sum to 1

    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    # Worker socket paths must fit AF_UNIX limits, unlike the inherited long TMPDIR.
    with tempfile.TemporaryDirectory(prefix="dogbreed-", dir=os.environ.get("WORKER_TMP_ROOT", "/tmp")) as worker_tmp:
        for variable in ("TMPDIR", "TEMP", "TMP"):
            os.environ[variable] = worker_tmp
        tempfile.tempdir = None
        torch.set_num_threads(CPU_THREADS)
        with threadpool_limits(limits=CPU_THREADS), Tracking() as tracking:
            main(tracking)
