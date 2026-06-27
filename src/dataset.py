import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def augment(patch, mode):
    """Eight-fold augmentation: identity, flips and 90-degree rotations."""
    if mode == 0:
        return patch
    elif mode == 1:
        return np.flipud(patch)
    elif mode == 2:
        return np.rot90(patch)
    elif mode == 3:
        return np.flipud(np.rot90(patch))
    elif mode == 4:
        return np.rot90(patch, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(patch, k=2))
    elif mode == 6:
        return np.rot90(patch, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(patch, k=3))


class DenoisingDataset(Dataset):
    """Generates (noisy_patch, noise) pairs on the fly.

    Patches are cropped at random positions from the training images, augmented,
    and corrupted with additive white Gaussian noise of standard deviation sigma.
    The target is the noise itself, so the model can be trained with the
    residual-learning objective.

    `length` controls how many samples make up one epoch (iterations * batch_size).
    """

    def __init__(self, image_dir, patch_size=40, sigma=25, length=1600 * 128):
        paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if not paths:
            raise FileNotFoundError(f'no .png images found in {image_dir}')
        self.images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        self.patch_size = patch_size
        self.sigma = sigma
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.images[np.random.randint(len(self.images))]
        h, w = img.shape
        ps = self.patch_size
        i = np.random.randint(0, h - ps + 1)
        j = np.random.randint(0, w - ps + 1)

        patch = img[i:i + ps, j:j + ps]
        patch = augment(patch, np.random.randint(0, 8)).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.sigma / 255.0, patch.shape).astype(np.float32)

        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)
        return to_tensor(patch + noise), to_tensor(noise)
