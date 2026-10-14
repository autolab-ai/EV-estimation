"""End-to-end fine-tuning of ResNet-50 for dog-breed-identification.

Replaces the frozen-feature + logistic-regression baseline: the same
torchvision ResNet-50 (IMAGENET1K_V1) backbone is fine-tuned end-to-end on the
120-class training images with random-resized-crop / horizontal-flip
augmentation and focal loss (gamma=2, no alpha weighting) instead of
label-smoothed cross-entropy, and softmax probabilities are predicted
directly. Test inference uses the same Resize(256)/CenterCrop(224) eval
transform as before.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os

# Must be set before torchvision is imported: pretrained weights are cached here
# instead of in the (quota-limited) login-node home directory.
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")

import contextlib

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "0.01"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
TRAIN_LIMIT = int(os.environ.get("TRAIN_LIMIT", "0"))  # 0 = all, smoke-test knob

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRAIN_TFM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**NORM),
])
EVAL_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(**NORM),
])


class FocalLoss(torch.nn.Module):
    """Focal loss (no alpha class weighting): (1 - p_t)**gamma * CE, mean over batch."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits.float(), targets,
                                               reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


class TrainSet(Dataset):
    """Labeled train images -> (image, class_index into the breeds column order)."""

    def __init__(self, root, ids, targets):
        self.root = root
        self.ids = list(ids)
        self.targets = list(targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            x = TRAIN_TFM(img.convert("RGB"))
        return x, self.targets[i]


class TestSet(Dataset):
    """Loads <root>/<id>.jpg for a list of ids, in the given order."""

    def __init__(self, root, ids):
        self.root = root
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, f"{self.ids[i]}.jpg")
        with Image.open(path) as img:
            return EVAL_TFM(img.convert("RGB"))


def _mlflow_run():
    """Attach to the preconfigured run; no-op context if tracking is unavailable."""
    try:
        import mlflow
        return mlflow, mlflow.start_run()
    except Exception as e:  # tracking must never kill training
        print(f"mlflow disabled: {e}")
        return None, contextlib.nullcontext()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    test_ids = sample["id"].tolist()
    # class_index = position of the breed in the sample-submission columns, so
    # softmax output column i already matches submission column order.
    breed_to_idx = {b: i for i, b in enumerate(breeds)}
    targets = labels["breed"].map(breed_to_idx).values
    train_ids = labels["id"].values
    if TRAIN_LIMIT:
        train_ids, targets = train_ids[:TRAIN_LIMIT], targets[:TRAIN_LIMIT]
    print(f"train={len(train_ids)} test={len(test_ids)} breeds={len(breeds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Linear(2048, len(breeds))
    net.to(device)

    loader = DataLoader(
        TrainSet(os.path.join(PUBLIC, "train"), train_ids, targets),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    steps_per_epoch = len(loader)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          nesterov=True, weight_decay=WEIGHT_DECAY)
    loss_fn = FocalLoss(gamma=2.0)
    # one linear-warmup epoch, then cosine decay to zero over the rest
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01,
                                              total_iters=steps_per_epoch),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, (EPOCHS - 1) * steps_per_epoch)),
        ],
        milestones=[steps_per_epoch],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    mlflow, run_ctx = _mlflow_run()
    log_failures = [0]

    def log_metric(name, value, step):
        if mlflow is None or log_failures[0]:
            return
        try:
            mlflow.log_metric(name, value, step=step)
        except Exception as e:
            log_failures[0] += 1
            print(f"mlflow logging disabled: {e}")

    with run_ctx:
        if mlflow is not None:
            try:
                mlflow.log_params({
                    "model": "resnet50_imagenet1k_v1_finetuned",
                    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "loss": "focal_gamma2.0",
                    "optimizer": "sgd_nesterov_m0.9",
                    "schedule": "1ep_linear_warmup+cosine",
                    "augmentation": "rrc224_s0.5-1.0+hflip",
                    "precision": "fp16_autocast",
                })
            except Exception as e:
                print(f"mlflow param logging failed: {e}")

        for epoch in range(EPOCHS):
            net.train()
            total, correct, seen = 0.0, 0, 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = net(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                seen += len(y)
            print(f"epoch {epoch + 1}/{EPOCHS} loss={total / seen:.4f} "
                  f"acc={correct / seen:.4f} lr={sched.get_last_lr()[0]:.5f}",
                  flush=True)
            log_metric("train_loss", total / seen, step=epoch)
            log_metric("train_acc", correct / seen, step=epoch)

        # inference over the test set with the eval transform
        net.eval()
        test_loader = DataLoader(
            TestSet(os.path.join(PUBLIC, "test"), test_ids),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        probs = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = net(x)
                probs.append(torch.softmax(logits.float(), dim=1).cpu())
        proba = torch.cat(probs).numpy().astype(np.float64)
        proba /= proba.sum(axis=1, keepdims=True)  # rows must sum to 1 exactly

    sub = pd.DataFrame(proba, columns=breeds)
    sub.insert(0, "id", test_ids)
    sub.to_csv("submission.csv", index=False)
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False)

    # graded by the hill's evaluator on submission/; no in-run grading


if __name__ == "__main__":
    main()
