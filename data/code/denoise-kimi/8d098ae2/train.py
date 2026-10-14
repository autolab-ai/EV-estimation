"""Baseline for denoising-dirty-documents.

Patch-regression CNN: a 4-level U-Net (base width 64) is trained with the
Zhao et al. (2017) composite restoration loss 0.84*(1-SSIM) + 0.16*L1
(see ssim.py) on random 128x128 crops of the (noisy, clean) training pairs
and applied to whole test images with 8-fold dihedral test-time augmentation.
Same recipe as the baseline otherwise. See README.md for details.

Writes submission.csv and submission/submission.csv; the hill's evaluator grades the latter.
"""

import os
import re
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ssim import SSIM

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
BASE_CH = cfg_int("BASE_CH", 64)
N_VAL = cfg_int("N_VAL", 10)
K_SYNTH = cfg_int("K_SYNTH", 6)
TTA = cfg_int("TTA", 1)


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


def make_noisy(clean, rng):
    """Corrupt a clean image with document-like artifacts.

    (a) gray ellipse stains, alpha-blended, then a Gaussian blur;
    (b) a smooth low-frequency multiplicative shading field;
    (c) sparse dark speckle. Returns a float32 image clipped to [0, 1].
    """
    img = clean.copy()
    h, w = img.shape
    # (a) stains
    for _ in range(int(rng.integers(4, 13))):  # UniformInteger(4, 12) inclusive
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ax = (max(int(rng.uniform(8, 80)) // 2, 1),
              max(int(rng.uniform(8, 80)) // 2, 1))
        val, alpha = float(rng.uniform(0.35, 0.85)), float(rng.uniform(0.25, 0.7))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), ax, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        img = img * (1.0 - alpha * mask) + val * (alpha * mask)
    sigma = float(rng.uniform(1, 4))
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # (b) shading
    field = rng.uniform(0.85, 1.0, size=(4, 4)).astype(np.float32)
    img = img * cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    # (c) speckle
    sp = rng.random(img.shape, dtype=np.float32) < 0.005
    img[sp] = rng.uniform(0.0, 0.5, size=int(sp.sum())).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


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
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.e1 = block(1, ch)
        self.e2 = block(ch, ch * 2)
        self.e3 = block(ch * 2, ch * 4)
        self.e4 = block(ch * 4, ch * 8)
        self.d3 = block(ch * 8 + ch * 4, ch * 4)
        self.d2 = block(ch * 4 + ch * 2, ch * 2)
        self.d1 = block(ch * 2 + ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        u3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], 1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], 1))
        return torch.sigmoid(self.out(d1))


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


@torch.no_grad()
def predict_tta(model, img, device, amp):
    """8-fold dihedral test-time augmentation.

    Run the network on all 4 rotations of the image, with and without a
    horizontal flip, invert each transform on the output, and average the
    8 predictions. The U-Net is fully convolutional, so these views are
    symmetry-equivalent; averaging them reduces prediction variance.
    """
    h, w = img.shape
    ph, pw = (-h) % 8, (-w) % 8
    t = torch.from_numpy(img)[None, None].to(device)
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    acc = torch.zeros_like(t, dtype=torch.float32)
    for flip in (False, True):
        tf = torch.flip(t, dims=[3]) if flip else t
        for k in range(4):
            tk = torch.rot90(tf, k, dims=[2, 3])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(tk)
            out = torch.rot90(out.float(), -k, dims=[2, 3])
            if flip:
                out = torch.flip(out, dims=[3])
            acc += out
    out = (acc / 8.0)[0, 0, :h, :w]
    return out.clamp_(0, 1).cpu().numpy()


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device={device}")

    tracker = Tracker()
    tracker.log_params({
        "model": "unet4", "base_ch": BASE_CH, "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH, "batch_size": BATCH_SIZE,
        "patch": PATCH, "lr": LR, "seed": SEED, "n_val": N_VAL,
        "k_synth": K_SYNTH,
        "loss": "0.84*(1-ssim)+0.16*l1", "optimizer": "adam",
        "sched": "cosine", "tta": TTA,
    })

    names, xs, ys = load_pairs()
    n_val = min(N_VAL, len(names) - 1)
    tr_x, tr_y = xs[:len(xs) - n_val], ys[:len(ys) - n_val]
    va_x, va_y = xs[len(xs) - n_val:], ys[len(ys) - n_val:]
    print(f"train={len(tr_x)} val={len(va_x)} shapes={sorted({a.shape for a in xs})}")

    # one-time synthetic augmentation: corrupt each clean training target K times
    gen_rng = np.random.default_rng(SEED + 1000)
    t_gen = time.time()
    syn_x, syn_y = [], []
    for clean in tr_y:
        for _ in range(K_SYNTH):
            syn_x.append(make_noisy(clean, gen_rng))
            syn_y.append(clean)
    tr_x = tr_x + syn_x
    tr_y = tr_y + syn_y
    print(f"added {len(syn_x)} synthetic pairs total_train={len(tr_x)} "
          f"gen_time={time.time() - t_gen:.0f}s", flush=True)
    tracker.log_metric("n_synthetic", len(syn_x))

    model = UNet(BASE_CH).to(device).to(memory_format=torch.channels_last)
    ssim = SSIM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * STEPS_PER_EPOCH
    )

    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        running_mse = 0.0
        for _ in range(STEPS_PER_EPOCH):
            bx, by = sample_batch(tr_x, tr_y, rng, BATCH_SIZE, PATCH)
            bx = bx.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            by = by.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                p = model(bx)
            # Zhao et al. (2017) composite restoration loss; float32 for
            # numerical stability under bf16 autocast.
            p = p.float()
            loss = 0.84 * (1.0 - ssim(p, by)) + 0.16 * F.l1_loss(p, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            running_mse += F.mse_loss(p, by).item()
            step += 1
        train_loss = running / STEPS_PER_EPOCH
        train_rmse = (running_mse / STEPS_PER_EPOCH) ** 0.5

        model.eval()
        se, cnt = 0.0, 0
        for a, b in zip(va_x, va_y):
            p = predict(model, a, device, amp)
            se += float(((p - b) ** 2).sum())
            cnt += b.size
        val_rmse = (se / max(cnt, 1)) ** 0.5
        print(f"epoch {epoch}/{EPOCHS} train_loss={train_loss:.5f} "
              f"train_rmse={train_rmse:.5f} "
              f"val_rmse={val_rmse:.5f} t={time.time() - t0:.0f}s", flush=True)
        tracker.log_metric("train_loss", train_loss, step=epoch)
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
        p = predict_tta(model, img, device, amp) if TTA else predict(model, img, device, amp)
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
