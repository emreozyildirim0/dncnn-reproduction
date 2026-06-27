import os
import glob

import cv2
import numpy as np
import torch


def calculate_psnr(a, b, data_range=1.0):
    """PSNR between two images given in the same range (default [0, 1])."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(data_range ** 2 / mse)


@torch.no_grad()
def evaluate(model, test_dir, sigma=25, device='cpu', seed=0):
    """Average PSNR of the denoised outputs over a test folder.

    The noise is generated with a fixed seed so the measurement is
    reproducible and comparable across runs.
    """
    model.eval()
    psnrs = []
    for path in sorted(glob.glob(os.path.join(test_dir, '*.png'))):
        clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        np.random.seed(seed)
        noisy = clean + np.random.normal(0, sigma / 255.0, clean.shape).astype(np.float32)
        x = torch.from_numpy(noisy).view(1, 1, *clean.shape).to(device)
        denoised = (x - model(x)).clamp(0.0, 1.0).cpu().numpy().squeeze()
        psnrs.append(calculate_psnr(denoised, clean))
    return float(np.mean(psnrs))
