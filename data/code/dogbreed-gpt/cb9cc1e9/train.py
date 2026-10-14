"""Initialize a breed head from frozen features, then tune the last bottleneck.

The evaluator alone grades submission/submission.csv. No in-run grading.
"""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

CONFIG = json.loads(Path(os.environ.get("DOGBREED_CONFIG", "config.json")).read_text())
# Keep CPU work inside the allocation; callers may override thread settings.
os.environ.setdefault("OMP_NUM_THREADS", str(CONFIG["num_threads"]))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CONFIG["num_threads"]))
os.environ.setdefault("MKL_NUM_THREADS", str(CONFIG["num_threads"]))

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

SEED = 0
TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Tracking:
    """Best-effort telemetry, attaching to the run supplied by the runner."""

    def __init__(self):
        self.client = None
        self.warned = False

    def disable(self, exc):
        self.client = None
        if not self.warned:
            print(f"Warning: tracking disabled: {exc}", flush=True)
            self.warned = True

    @contextmanager
    def run(self):
        active = None
        try:
            import mlflow
            active = mlflow.start_run()
            self.client = mlflow
        except Exception as exc:
            self.disable(exc)
        try:
            yield self
        finally:
            if active is not None:
                try:
                    active.__exit__(None, None, None)
                except Exception as exc:
                    self.disable(exc)

    def params(self, values):
        if self.client is not None:
            try:
                self.client.log_params(values)
            except Exception as exc:
                self.disable(exc)

    def metric(self, name, value, step=0):
        if self.client is not None:
            try:
                self.client.log_metric(name, float(value), step=step)
            except Exception as exc:
                self.disable(exc)


@contextmanager
def worker_tempdir():
    # Multiprocessing's Unix sockets cannot use the runner's long temp paths.
    keys = ("TMPDIR", "TMP", "TEMP")
    old_env = {key: os.environ.get(key) for key in keys}
    old_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="dog-", dir=CONFIG["worker_temp_root"]) as path:
        for key in keys:
            os.environ[key] = path
        tempfile.tempdir = path
        try:
            yield
        finally:
            tempfile.tempdir = old_tempdir
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class ImageFolderList(Dataset):
    def __init__(self, root, ids):
        self.root = root
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(os.path.join(self.root, f"{self.ids[i]}.jpg")) as img:
            return TFM(img.convert("RGB"))


def image_loader(root, ids, device):
    return DataLoader(
        ImageFolderList(root, ids), batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def extract_training(net, root, ids, device):
    """Cache exactly the input to layer4[-1], while keeping ordinary features."""
    cache = torch.empty((len(ids), 2048, 7, 7), dtype=torch.float16, device="cpu")
    features = np.empty((len(ids), 2048), dtype=np.float32)
    cursor = 0

    def save_input(module, inputs):
        activations = inputs[0]
        assert activations.shape[1:] == (2048, 7, 7)
        cache[cursor:cursor + len(activations)].copy_(activations.detach())

    hook = net.layer4[-1].register_forward_pre_hook(save_input)
    try:
        for batch_index, batch in enumerate(image_loader(root, ids, device)):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                # Includes the last bottleneck, avgpool and the identity fc.
                out = net(batch)
            features[cursor:cursor + len(batch)] = out.float().cpu().numpy()
            cursor += len(batch)
            if (batch_index + 1) % 25 == 0:
                print(f"Training feature extraction: {cursor}/{len(ids)} images", flush=True)
    finally:
        hook.remove()
    assert cursor == len(ids)
    return features, cache


def initialize_head(net, clf, device):
    """Rows retain clf.classes_ order, including during supervised training."""
    net.requires_grad_(False)
    head = torch.nn.Linear(2048, len(clf.classes_)).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(clf.coef_, dtype=torch.float32, device=device))
        head.bias.copy_(torch.as_tensor(clf.intercept_, dtype=torch.float32, device=device))
    net.fc = head
    block = net.layer4[-1]
    convolution_params = []
    for convolution in (block.conv1, block.conv2, block.conv3):
        convolution.requires_grad_(True)
        convolution_params.extend(convolution.parameters())
    # Eval mode leaves convolution gradients enabled but fixes every BN statistic.
    net.eval()
    return convolution_params


def fine_tune(net, cache, targets, convolution_params, device, tracking):
    loader = DataLoader(
        TensorDataset(cache, targets), batch_size=CONFIG["finetune_batch_size"],
        shuffle=True, generator=torch.Generator().manual_seed(SEED),
        num_workers=0, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.SGD([
        {"params": convolution_params, "lr": 0.0001},
        {"params": net.fc.parameters(), "lr": 0.001},
    ], momentum=0.9, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    net.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    batches = 0
    for epoch in range(2):
        loss_sum, seen = 0.0, 0
        for activations, target in loader:
            if time.monotonic() - started >= 90.0:
                break
            activations = activations.to(
                device, dtype=torch.float16 if device.type == "cuda" else torch.float32,
                non_blocking=True,
            )
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = net.layer4[-1](activations)
                logits = net.fc(net.avgpool(out).flatten(1))
                loss = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item() * len(target)
            seen += len(target)
            batches += 1
        elapsed = time.monotonic() - started
        if seen:
            mean_loss = loss_sum / seen
            print(f"Fine-tuning epoch {epoch + 1}: images={seen} loss={mean_loss:.6f} "
                  f"elapsed={elapsed:.1f}s batches={batches}", flush=True)
            tracking.metric("train_loss", mean_loss, epoch + 1)
            tracking.metric("finetune_images", seen, epoch + 1)
            tracking.metric("finetune_seconds", elapsed, epoch + 1)
            tracking.metric("finetune_batches", batches, epoch + 1)
        if elapsed >= 90.0:
            break
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def predict(net, root, ids, device):
    net.eval()
    probabilities = []
    for batch in image_loader(root, ids, device):
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = net(batch)
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def main(tracking):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(CONFIG["num_threads"])
    data_root = Path(os.environ.get("EV_DATA_DIR", CONFIG["data_dir"]))
    public = Path(os.environ.get("DOGBREED_PUBLIC_DIR", data_root / "dog-breed-identification" / "prepared" / "public"))
    labels = pd.read_csv(public / "labels.csv")
    sample = pd.read_csv(public / "sample_submission.csv")
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)} device={device}", flush=True)
    tracking.params({
        "model": "resnet50", "weights": "IMAGENET1K_V1", "resize": 256, "crop": 224,
        "seed": SEED, "batch_size": CONFIG["batch_size"], "num_workers": CONFIG["num_workers"],
        "classifier_C": 0.03, "classifier_max_iter": 1000,
        "finetune_batch_size": CONFIG["finetune_batch_size"], "finetune_epochs": 2,
        "finetune_max_seconds": 90, "conv_lr": 0.0001, "head_lr": 0.001,
        "momentum": 0.9, "weight_decay": 0.0001, "batchnorm_frozen": True,
        "train_images": len(labels), "test_images": len(test_ids), "num_classes": len(breeds),
    })

    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.requires_grad_(False).eval().to(device)
    started = time.monotonic()
    x_train, cache = extract_training(net, public / "train", labels["id"], device)
    tracking.metric("feature_seconds", time.monotonic() - started)
    print(f"Training features ready; cached {len(cache)} bottleneck inputs", flush=True)

    clf = LogisticRegression(C=0.03, max_iter=1000, random_state=SEED)
    started = time.monotonic()
    clf.fit(x_train, labels["breed"].values)
    tracking.metric("classifier_seconds", time.monotonic() - started)
    classes = list(clf.classes_)
    assert len(classes) == len(breeds) == 120
    convolution_params = initialize_head(net, clf, device)
    class_to_index = {breed: index for index, breed in enumerate(classes)}
    targets = torch.tensor([class_to_index[b] for b in labels["breed"]], dtype=torch.long)
    del x_train, clf

    print("Starting fine-tuning: at most two epochs or 90 seconds", flush=True)
    fine_tune(net, cache, targets, convolution_params, device, tracking)
    del cache, targets, convolution_params

    print("Running one forward pass per test image", flush=True)
    started = time.monotonic()
    proba = predict(net, public / "test", test_ids, device)
    tracking.metric("inference_seconds", time.monotonic() - started)
    order = [classes.index(b) for b in breeds]
    proba = proba[:, order]
    proba /= proba.sum(axis=1, keepdims=True)
    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)
    sub.to_csv("submission/submission.csv", index=False)
    print("Wrote submission/submission.csv", flush=True)


if __name__ == "__main__":
    with worker_tempdir(), Tracking().run() as tracking:
        main(tracking)
