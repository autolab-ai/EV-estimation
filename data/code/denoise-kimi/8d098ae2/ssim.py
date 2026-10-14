"""SSIM (structural similarity) for image-restoration losses.

Standard Wang et al. (2004) SSIM with an 11x11 Gaussian window (sigma 1.5),
as used by Zhao et al. (2017, "Loss Functions for Image Restoration with
Neural Networks") in their composite SSIM+L1 restoration loss. Inputs are
single-channel Nx1xHxW float tensors with data range 1.0 (the network output
already passes through a sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size=11, sigma=1.5):
    """Fixed 2D Gaussian window normalized to sum 1, shape 1x1xHxW."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


class SSIM(nn.Module):
    """Mean SSIM over the batch, computed with F.conv2d local statistics."""

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.register_buffer("window", gaussian_window(window_size, sigma))

    def forward(self, p, y):
        w = self.window
        mu_p = F.conv2d(p, w)
        mu_y = F.conv2d(y, w)
        mu_p2, mu_y2, mu_py = mu_p * mu_p, mu_y * mu_y, mu_p * mu_y
        sigma_p2 = F.conv2d(p * p, w) - mu_p2
        sigma_y2 = F.conv2d(y * y, w) - mu_y2
        sigma_py = F.conv2d(p * y, w) - mu_py
        ssim_map = ((2.0 * mu_py + self.C1) * (2.0 * sigma_py + self.C2)) / (
            (mu_p2 + mu_y2 + self.C1) * (sigma_p2 + sigma_y2 + self.C2)
        )
        return ssim_map.mean()
