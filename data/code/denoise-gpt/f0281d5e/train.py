"""Compact NAFNet for denoising-dirty-documents.

Patch-regression CNN: a residual NAFNet is trained with MSE loss on random
128x128 crops of the (noisy, clean) training pairs and applied to whole test
images. See README.md for details.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import json
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

COMP = "denoising-dirty-documents"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")


def cfg_int(name, default):
    return int(os.environ.get(name, default))


def cfg_float(name, default):
    return float(os.environ.get(name, default))


SEED = cfg_int("SEED", 0)
EPOCHS = cfg_int("EPOCHS", 20)
STEPS_PER_EPOCH = cfg_int("STEPS_PER_EPOCH", 100)
BATCH_SIZE = cfg_int("BATCH_SIZE", 32)
PATCH = cfg_int("PATCH", 128)
LR = cfg_float("LR", 2e-3)
BASE_CH = cfg_int("BASE_CH", 16)
N_VAL = cfg_int("N_VAL", 10)


# ---------------------------------------------------------------- mlflow (soft)
class Tracker:
    """Best-effort MLflow logging: disables itself on the first failure."""

    def __init__(self):
        self.ok = False
        self.ctx = None
        try:
            import mlflow

            self.mlflow = mlflow
            self.ctx = mlflow.start_run()
            self.ctx.__enter__()
            self.ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[mlflow] disabled: {exc}")

    def log_params(self, params):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(params)
        except Exception as exc:
            print(f"[mlflow] disabled: {exc}")
            self.ok = False

    def log_metric(self, key, value, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(key, float(value), step=step)
        except Exception as exc:
            print(f"[mlflow] disabled: {exc}")
            self.ok = False

    def close(self):
        if self.ctx is None:
            return
        try:
            self.ctx.__exit__(None, None, None)
        except Exception:
            pass


# ---------------------------------------------------------------------- dataset
def load_gray(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def load_pairs():
    names = sorted(os.listdir(os.path.join(PUBLIC, "train")))
    xs, ys = [], []
    for n in names:
        xs.append(load_gray(os.path.join(PUBLIC, "train", n)))
        ys.append(load_gray(os.path.join(PUBLIC, "train_cleaned", n)))
    return names, xs, ys


def sample_batch(xs, ys, rng, batch, patch):
    bx = np.empty((batch, 1, patch, patch), dtype=np.float32)
    by = np.empty((batch, 1, patch, patch), dtype=np.float32)
    for i in range(batch):
        k = rng.integers(len(xs))
        h, w = xs[k].shape
        r = rng.integers(h - patch + 1)
        c = rng.integers(w - patch + 1)
        px = xs[k][r:r + patch, c:c + patch]
        py = ys[k][r:r + patch, c:c + patch]
        if rng.random() < 0.5:
            px, py = px[:, ::-1], py[:, ::-1]
        if rng.random() < 0.5:
            px, py = px[::-1], py[::-1]
        bx[i, 0] = px
        by[i, 0] = py
    return torch.from_numpy(bx), torch.from_numpy(by)


# ------------------------------------------------------------------------ model
class LayerNorm2d(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.expand1 = nn.Conv2d(channels, 2 * channels, 1, bias=True)
        self.depthwise = nn.Conv2d(
            2 * channels, 2 * channels, 3, padding=1,
            groups=2 * channels, bias=True,
        )
        self.gate1 = SimpleGate()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.project1 = nn.Conv2d(channels, channels, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.expand2 = nn.Conv2d(channels, 2 * channels, 1, bias=True)
        self.gate2 = SimpleGate()
        self.project2 = nn.Conv2d(channels, channels, 1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        g = self.gate1(self.depthwise(self.expand1(self.norm1(x))))
        g = g * self.attention(g)
        y = x + self.beta * self.project1(g)
        g = self.project2(self.gate2(self.expand2(self.norm2(y))))
        return y + self.gamma * g


class NAFNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.intro = nn.Conv2d(1, ch, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        width = ch
        for _ in range(3):
            self.encoders.append(NAFBlock(width))
            self.downs.append(nn.Conv2d(width, width * 2, 2, stride=2, bias=True))
            width *= 2

        self.middle = nn.Sequential(*(NAFBlock(width) for _ in range(4)))
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(3):
            self.ups.append(nn.Sequential(
                nn.Conv2d(width, width * 2, 1, bias=True),
                nn.PixelShuffle(2),
            ))
            width //= 2
            self.decoders.append(NAFBlock(width))
        self.ending = nn.Conv2d(ch, 1, 3, padding=1, bias=True)

    def forward(self, dirty):
        x = self.intro(dirty)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(up(x) + skip)
        return dirty + self.ending(x)


@torch.no_grad()
def predict(model, img, device, amp):
    """Full-image inference; reflect-pad to a multiple of 8."""
    h, w = img.shape
    ph, pw = (-h) % 8, (-w) % 8
    t = torch.from_numpy(img)[None, None].to(device)
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = model(t)
    return out.float()[0, 0, :h, :w].clamp_(0, 1).cpu().numpy()


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    config_path = os.environ.get(
        "RUNTIME_CONFIG", os.path.join(os.path.dirname(__file__), "runtime.json")
    )
    with open(config_path) as f:
        runtime = json.load(f)
    threads = int(os.environ.get("OMP_NUM_THREADS", runtime["num_threads"]))
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    torch.set_num_threads(max(1, min(threads, available or threads)))
    device_name = os.environ.get("DENOISE_DEVICE", runtime["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    amp = device.type == "cuda"
    print(f"device={device} threads={torch.get_num_threads()}")

    tracker = Tracker()
    tracker.log_params({
        "model": "nafnet", "base_ch": BASE_CH, "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH, "batch_size": BATCH_SIZE,
        "patch": PATCH, "lr": LR, "seed": SEED, "n_val": N_VAL,
        "loss": "mse", "optimizer": "adam", "sched": "cosine",
    })

    names, xs, ys = load_pairs()
    n_val = min(N_VAL, len(names) - 1)
    tr_x, tr_y = xs[:len(xs) - n_val], ys[:len(ys) - n_val]
    va_x, va_y = xs[len(xs) - n_val:], ys[len(ys) - n_val:]
    print(f"train={len(tr_x)} val={len(va_x)} shapes={sorted({a.shape for a in xs})}")

    model = NAFNet(BASE_CH).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * STEPS_PER_EPOCH
    )

    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for _ in range(STEPS_PER_EPOCH):
            bx, by = sample_batch(tr_x, tr_y, rng, BATCH_SIZE, PATCH)
            bx = bx.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            by = by.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.mse_loss(model(bx), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
        train_rmse = (running / STEPS_PER_EPOCH) ** 0.5

        model.eval()
        se, cnt = 0.0, 0
        for a, b in zip(va_x, va_y):
            p = predict(model, a, device, amp)
            se += float(((p - b) ** 2).sum())
            cnt += b.size
        val_rmse = (se / max(cnt, 1)) ** 0.5
        print(f"epoch {epoch}/{EPOCHS} train_rmse={train_rmse:.5f} "
              f"val_rmse={val_rmse:.5f} t={time.time() - t0:.0f}s", flush=True)
        tracker.log_metric("train_rmse", train_rmse, step=epoch)
        tracker.log_metric("val_rmse", val_rmse, step=epoch)
        tracker.log_metric("lr", sched.get_last_lr()[0], step=epoch)

    # ------------------------------------------------------------- predictions
    model.eval()
    sample = pd.read_csv(os.path.join(PUBLIC, "sampleSubmission.csv"))
    ids, vals = [], []
    for fn in sorted(os.listdir(os.path.join(PUBLIC, "test"))):
        stem = os.path.splitext(fn)[0]
        img = load_gray(os.path.join(PUBLIC, "test", fn))
        p = predict(model, img, device, amp)
        h, w = p.shape
        rows = np.repeat(np.arange(1, h + 1), w)
        cols = np.tile(np.arange(1, w + 1), h)
        ids.append(np.char.add(np.char.add(
            np.full(h * w, stem + "_"), rows.astype(str)),
            np.char.add(np.full(h * w, "_"), cols.astype(str))))
        vals.append(p.reshape(-1))
    pred = pd.Series(np.concatenate(vals), index=np.concatenate(ids))
    print(f"predicted pixels={len(pred)} expected={len(sample)}")

    aligned = pred.reindex(sample["id"].values)
    missing = int(aligned.isna().sum())
    if missing:
        raise SystemExit(f"{missing} submission ids have no prediction")
    sub = pd.DataFrame({"id": sample["id"].values,
                        "value": np.clip(aligned.values, 0.0, 1.0)})
    sub.to_csv("submission.csv", index=False, float_format="%.5f")
    os.makedirs('submission', exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join('submission', 'submission.csv'), index=False, float_format="%.5f")
    print(f"wrote submission.csv rows={len(sub)} t={time.time() - t0:.0f}s", flush=True)

    # graded by the hill's evaluator on submission/; no in-run grading
    tracker.close()


if __name__ == "__main__":
    main()
