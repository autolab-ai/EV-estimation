"""Dog Breed Identification: end-to-end fine-tuning of ConvNeXt-Tiny (ImageNet-22k).

Fine-tunes timm's `convnext_tiny.fb_in22k_ft_in1k` on the 120 breeds with a
stratified 90/10 train/validation split, AdamW (backbone lr 1e-4, head lr 1e-3),
linear warmup + cosine decay, label smoothing, bf16/fp16 autocast, horizontal-flip
test-time augmentation and temperature scaling picked on the validation split.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import tempfile

# Pretrained weights cache on scratch (set before importing timm/torch hub).
os.environ.setdefault("HF_HOME", "<HF_HOME>")
os.environ.setdefault("TORCH_HOME", "<TORCH_HOME>")
# Bound MLflow retry/backoff so an unreachable tracking server costs seconds, not minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
# The workspace TMPDIR can exceed the 108-byte AF_UNIX limit, which kills
# DataLoader workers; use a short temp dir when available.
if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK):
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"

import contextlib
import math
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

T0 = time.time()

torch.multiprocessing.set_sharing_strategy("file_system")

COMP = "dog-breed-identification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
MODEL_NAME = os.environ.get("MODEL_NAME", "convnext_tiny.fb_in22k_ft_in1k")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
LR_MIN = 1e-6
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 100
LABEL_SMOOTHING = 0.1
DROP_PATH = 0.1
CLIP_NORM = 1.0
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", str(7 * 60)))
SMOKE = os.environ.get("SMOKE", "0") == "1"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TFM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.35, 1.0), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
EVAL_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Best-effort MLflow tracking: must never kill the run.
# ----------------------------------------------------------------------------
class _Tracker:
    def __init__(self):
        self.ok = False
        self.mlflow = None
        try:
            import mlflow  # noqa

            self.mlflow = mlflow
            self.ok = True
        except Exception as e:  # pragma: no cover
            print(f"[mlflow] unavailable: {e}", flush=True)

    @contextlib.contextmanager
    def start(self):
        started = False
        if self.ok:
            try:
                self.mlflow.start_run()
                started = True
            except Exception as e:
                print(f"[mlflow] start_run failed, tracking disabled: {e}", flush=True)
                self.ok = False
        try:
            yield self
        finally:
            if started:
                try:
                    self.mlflow.end_run()
                except Exception as e:
                    print(f"[mlflow] end_run failed (ignored): {e}", flush=True)

    def log_params(self, d):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:
            print(f"[mlflow] log_params failed, tracking disabled: {e}", flush=True)
            self.ok = False

    def log_metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print(f"[mlflow] log_metric failed, tracking disabled: {e}", flush=True)
            self.ok = False


class DogDataset(Dataset):
    def __init__(self, root, ids, labels, tfm):
        self.root, self.ids, self.labels, self.tfm = root, list(ids), labels, tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(os.path.join(self.root, f"{self.ids[i]}.jpg")) as img:
            x = self.tfm(img.convert("RGB"))
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y


def make_loader(ds, shuffle, drop_last, workers):
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=drop_last,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )


@torch.no_grad()
def predict_logits(model, loader, device, amp_dtype):
    """Average logits of each image and its horizontal flip."""
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            l1 = model(x)
            l2 = model(torch.flip(x, dims=[3]))
        outs.append(((l1.float() + l2.float()) / 2).cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def nll(logits, y, T=1.0):
    return F.cross_entropy(logits / T, y).item()


def run_training(model, train_loader, val_loader, device, amp_dtype, use_scaler, tracker, epochs):
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    head_params = list(model.head.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )
    base_lrs = [LR_BACKBONE, LR_HEAD]

    def lr_at(step, base):
        if step < WARMUP_STEPS:
            return base * (step + 1) / WARMUP_STEPS
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return LR_MIN + 0.5 * (base - LR_MIN) * (1 + math.cos(math.pi * min(1.0, prog)))

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    step = 0
    for epoch in range(epochs):
        model.train()
        run_loss, n = 0.0, 0
        for x, y in train_loader:
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = lr_at(step, base)
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
            step += 1
            if SMOKE and step >= 3:
                break
        train_loss = run_loss / max(1, n)
        val_logits, val_y = predict_logits(model, val_loader, device, amp_dtype)
        val_ll = nll(val_logits, val_y)
        val_acc = (val_logits.argmax(1) == val_y).float().mean().item()
        elapsed = time.time() - T0
        log(f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_logloss={val_ll:.4f} val_acc={val_acc:.4f}")
        tracker.log_metric("train_loss", train_loss, step=epoch + 1)
        tracker.log_metric("val_logloss", val_ll, step=epoch + 1)
        tracker.log_metric("val_acc", val_acc, step=epoch + 1)
        if elapsed > TIME_BUDGET_S:
            log(f"time budget exceeded ({elapsed:.0f}s > {TIME_BUDGET_S:.0f}s); stopping training")
            break
        if SMOKE:
            break
    return model


def main():
    global BATCH_SIZE
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    labels = pd.read_csv(os.path.join(PUBLIC, "labels.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    breeds = [c for c in sample.columns if c != "id"]
    breed_idx = {b: i for i, b in enumerate(breeds)}
    test_ids = sample["id"].tolist()
    y_all = labels["breed"].map(breed_idx).values.astype(np.int64)
    log(f"train={len(labels)} test={len(test_ids)} breeds={len(breeds)}")

    tr_ids, va_ids, y_tr, y_va = train_test_split(
        labels["id"].astype(str).to_numpy(dtype=object), y_all, test_size=0.1, stratify=y_all, random_state=SEED)
    if SMOKE:
        tr_ids, y_tr, va_ids, y_va = tr_ids[:256], y_tr[:256], va_ids[:128], y_va[:128]
        test_ids_run = test_ids[:128]
    else:
        test_ids_run = test_ids
    log(f"split: train={len(tr_ids)} val={len(va_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype, use_scaler = torch.bfloat16, False
    else:
        amp_dtype, use_scaler = torch.float16, device.type == "cuda"
    log(f"device={device} amp={amp_dtype} scaler={use_scaler}")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(breeds), drop_path_rate=DROP_PATH)
    model = model.to(device).to(memory_format=torch.channels_last)
    log(f"model={MODEL_NAME} params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_root, test_root = os.path.join(PUBLIC, "train"), os.path.join(PUBLIC, "test")
    train_ds = DogDataset(train_root, tr_ids, y_tr, TRAIN_TFM)
    val_ds = DogDataset(train_root, va_ids, y_va, EVAL_TFM)
    test_ds = DogDataset(test_root, test_ids_run, None, EVAL_TFM)

    tracker = _Tracker()
    with tracker.start():
        tracker.log_params({
            "model": MODEL_NAME, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD, "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS, "label_smoothing": LABEL_SMOOTHING,
            "drop_path": DROP_PATH, "amp": str(amp_dtype), "seed": SEED,
        })

        workers = NUM_WORKERS
        while True:
            try:
                train_loader = make_loader(train_ds, True, True, workers)
                val_loader = make_loader(val_ds, False, False, workers)
                test_loader = make_loader(test_ds, False, False, workers)
                log(f"batch={BATCH_SIZE} workers={workers} steps/epoch={len(train_loader)}")
                run_training(model, train_loader, val_loader, device, amp_dtype, use_scaler, tracker, EPOCHS)
                break
            except torch.cuda.OutOfMemoryError:
                if BATCH_SIZE <= 32:
                    raise
                log("CUDA OOM at batch %d; retrying with batch 32" % BATCH_SIZE)
                BATCH_SIZE = 32
                torch.cuda.empty_cache()
                model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(breeds), drop_path_rate=DROP_PATH)
                model = model.to(device).to(memory_format=torch.channels_last)

        # Temperature scaling on the validation split.
        val_logits, val_y = predict_logits(model, val_loader, device, amp_dtype)
        temps = np.arange(0.5, 2.01, 0.05)
        losses = [nll(val_logits, val_y, float(T)) for T in temps]
        T_best = float(temps[int(np.argmin(losses))])
        log(f"temperature: T={T_best:.2f} val_logloss {nll(val_logits, val_y):.4f} -> {min(losses):.4f}")
        tracker.log_metric("val_logloss_calibrated", min(losses))
        tracker.log_metric("temperature", T_best)

        test_logits, _ = predict_logits(model, test_loader, device, amp_dtype)
        proba = F.softmax(test_logits / T_best, dim=1).double().numpy()
        proba /= proba.sum(axis=1, keepdims=True)

        sub = pd.DataFrame(proba, columns=breeds)
        sub.insert(0, "id", test_ids_run)
        sub.to_csv("submission.csv", index=False)
        os.makedirs("submission", exist_ok=True)  # the hill grades this directory
        sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
        log(f"wrote submission/submission.csv rows={len(sub)} total={time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
