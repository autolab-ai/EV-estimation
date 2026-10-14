"""Leaf-classification: logistic regression on the 192 descriptors blended 50/50 with a
small two-branch neural network (64x64 binary silhouette + standardised features).

Protocol:
  (a) stratified 80/20 split (random_state=0): print holdout log loss of the NN alone,
      the LR alone and the arithmetic 50/50 blend;
  (b) retrain the NN from scratch on all training rows, predict the test images with
      8-rotation test-time augmentation;
  (c) final probabilities = 0.5 * p_nn + 0.5 * p_lr (LR refit on all rows), clipped and
      row-normalised; written to submission.csv and submission/submission.csv.

A wall-clock guard breaks out of either training loop after 9 minutes and proceeds to
prediction so the run always finishes inside the evaluator's limit.
"""

import math
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

T0 = time.time()

COMP = "leaf-classification"
DATA_DIR = os.environ.get("EV_DATA_DIR", "<DATA_ROOT>")
PUBLIC = os.path.join(DATA_DIR, COMP, "prepared", "public")
SEED = 0
C = 1.0
IMG = 64
EPOCHS = int(os.environ.get("NN_EPOCHS", "80"))
BATCH = 32
LR = 1e-3
WD = 1e-4
THREADS = int(os.environ.get("NN_THREADS", "4"))
TIME_GUARD_S = float(os.environ.get("TIME_GUARD_S", str(9 * 60)))
BLEND_W = 0.5


# ----------------------------------------------------------------------------- mlflow
class _Tracker:
    """Best-effort MLflow wrapper: disables itself after the first failure."""

    def __init__(self):
        self.ok = False
        self.mlflow = None
        self._ctx = None
        try:
            import mlflow  # noqa

            self.mlflow = mlflow
            self._ctx = mlflow.start_run()
            self.ok = True
        except Exception as e:  # pragma: no cover
            print("mlflow disabled: %s" % e)

    def log_params(self, d):
        if not self.ok:
            return
        try:
            self.mlflow.log_params(d)
        except Exception as e:
            print("mlflow disabled: %s" % e)
            self.ok = False

    def log_metric(self, k, v, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print("mlflow disabled: %s" % e)
            self.ok = False

    def end(self):
        if self.ok:
            try:
                self.mlflow.end_run()
            except Exception:
                pass


# ----------------------------------------------------------------------------- data
def featurize(df):
    return df[[c for c in df.columns if c not in ("id", "species")]].to_numpy(dtype=np.float64)


def load_image(path):
    arr = np.asarray(Image.open(path).convert("L"))
    mask = arr > 127
    if mask.mean() > 0.5:  # background is white in the raw files: make the leaf 1
        mask = ~mask
    h, w = mask.shape
    s = max(h, w)
    sq = np.zeros((s, s), dtype=np.uint8)
    top, left = (s - h) // 2, (s - w) // 2
    sq[top:top + h, left:left + w] = mask.astype(np.uint8) * 255
    im = Image.fromarray(sq).resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_images(ids):
    x = np.stack([load_image(os.path.join(PUBLIC, "images", "%d.jpg" % i)) for i in ids])
    return torch.from_numpy(x).unsqueeze(1)  # (N, 1, 64, 64)


# ----------------------------------------------------------------------------- model
class TwoBranchNet(nn.Module):
    def __init__(self, n_feat=192, n_classes=99):
        super().__init__()
        self.img = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.feat = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(256, n_classes)

    def forward(self, img, feat):
        return self.head(torch.cat([self.img(img), self.feat(feat)], dim=1))


def train_nn(x_img, x_feat, y, n_classes, tracker, tag):
    torch.manual_seed(SEED)
    model = TwoBranchNet(x_feat.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    aug = T.Compose([T.RandomAffine(degrees=180, scale=(0.9, 1.1)), T.RandomHorizontalFlip(0.5)])
    loss_fn = nn.CrossEntropyLoss()
    n = len(y)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        if time.time() - T0 > TIME_GUARD_S:
            print("[%s] time guard hit at epoch %d, stopping training" % (tag, ep))
            break
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = aug(x_img[idx])
            fb = x_feat[idx] + 0.05 * torch.randn(len(idx), x_feat.shape[1], generator=g)
            out = model(xb, fb)
            loss = loss_fn(out, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        sched.step()
        tracker.log_metric("%s_train_loss" % tag, tot / cnt, step=ep)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print("[%s] epoch %d train_loss=%.4f elapsed=%.0fs" % (tag, ep, tot / cnt, time.time() - T0))
    return model


@torch.no_grad()
def predict_nn(model, x_img, x_feat, tta=False):
    model.eval()
    angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] if tta else [0.0]
    probs = torch.zeros(len(x_feat), model.head.out_features)
    for a in angles:
        xi = TF.rotate(x_img, a) if a != 0.0 else x_img
        for i in range(0, len(x_feat), 256):
            probs[i:i + 256] += F.softmax(model(xi[i:i + 256], x_feat[i:i + 256]), dim=1)
    probs /= len(angles)
    return probs.double().numpy()


# ----------------------------------------------------------------------------- main
def main():
    torch.set_num_threads(THREADS)
    tracker = _Tracker()
    tracker.log_params({"C": C, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                        "img": IMG, "blend_w": BLEND_W, "threads": THREADS})

    train = pd.read_csv(os.path.join(PUBLIC, "train.csv"))
    test = pd.read_csv(os.path.join(PUBLIC, "test.csv"))
    sample = pd.read_csv(os.path.join(PUBLIC, "sample_submission.csv"))
    classes = [c for c in sample.columns if c != "id"]
    cls_index = {c: i for i, c in enumerate(classes)}

    x_all, y_all = featurize(train), train["species"].values
    y_all_i = np.array([cls_index[s] for s in y_all])
    x_test = featurize(test)

    scaler = StandardScaler().fit(x_all)
    x_all_s, x_test_s = scaler.transform(x_all), scaler.transform(x_test)

    img_all = load_images(train["id"].values)
    img_test = load_images(test["id"].values)
    print("images loaded: train %s test %s in %.1fs" % (tuple(img_all.shape), tuple(img_test.shape), time.time() - T0))

    # ---- (a) holdout evaluation
    idx_tr, idx_val = train_test_split(np.arange(len(y_all)), test_size=0.2,
                                       stratify=y_all, random_state=SEED)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
    clf.fit(x_all_s[idx_tr], y_all[idx_tr])
    p_lr_val = np.zeros((len(idx_val), len(classes)))
    p_lr_val[:, [cls_index[c] for c in clf.classes_]] = clf.predict_proba(x_all_s[idx_val])

    model = train_nn(img_all[idx_tr], torch.from_numpy(x_all_s[idx_tr]).float(),
                     torch.from_numpy(y_all_i[idx_tr]).long(), len(classes), tracker, "holdout")
    p_nn_val = predict_nn(model, img_all[idx_val], torch.from_numpy(x_all_s[idx_val]).float())
    p_nn_val /= p_nn_val.sum(axis=1, keepdims=True)
    p_bl_val = BLEND_W * p_nn_val + (1 - BLEND_W) * p_lr_val
    labels = np.arange(len(classes))
    ll_lr = log_loss(y_all_i[idx_val], p_lr_val, labels=labels)
    ll_nn = log_loss(y_all_i[idx_val], p_nn_val, labels=labels)
    ll_bl = log_loss(y_all_i[idx_val], p_bl_val, labels=labels)
    print("local_holdout_logloss=%.5f" % ll_lr)
    print("holdout_logloss_nn=%.5f" % ll_nn)
    print("holdout_logloss_blend=%.5f" % ll_bl)
    tracker.log_metric("local_holdout_logloss", ll_lr)
    tracker.log_metric("holdout_logloss_nn", ll_nn)
    tracker.log_metric("holdout_logloss_blend", ll_bl)

    # ---- (b) full-data NN + TTA
    model = train_nn(img_all, torch.from_numpy(x_all_s).float(),
                     torch.from_numpy(y_all_i).long(), len(classes), tracker, "full")
    p_nn = predict_nn(model, img_test, torch.from_numpy(x_test_s).float(), tta=True)

    # ---- (c) LR refit on all rows and blend
    clf.fit(x_all_s, y_all)
    p_lr = np.zeros((len(x_test_s), len(classes)))
    p_lr[:, [cls_index[c] for c in clf.classes_]] = clf.predict_proba(x_test_s)
    proba = BLEND_W * p_nn + (1 - BLEND_W) * p_lr
    proba = np.clip(proba, 1e-15, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)

    sub = pd.concat([pd.DataFrame({"id": test["id"].values}),
                     pd.DataFrame(proba, columns=classes)], axis=1)
    sub.to_csv("submission.csv", index=False)
    os.makedirs("submission", exist_ok=True)  # the hill grades this directory
    sub.to_csv(os.path.join("submission", "submission.csv"), index=False)
    print("done in %.1fs" % (time.time() - T0))
    tracker.log_metric("run_seconds", time.time() - T0)
    tracker.end()


if __name__ == "__main__":
    main()
