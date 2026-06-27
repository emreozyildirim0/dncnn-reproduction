import os
import glob
import argparse

import cv2
import numpy as np
import torch

from model import DnCNN
from utils import calculate_psnr


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a trained DnCNN model on a test set.')
    p.add_argument('--checkpoint', required=True, help='path to the trained weights (.pth)')
    p.add_argument('--test_dir', required=True, help='folder with clean test images (e.g. Set12, BSD68)')
    p.add_argument('--sigma', type=int, default=25)
    p.add_argument('--depth', type=int, default=17)
    p.add_argument('--save_dir', default=None, help='optional folder to write the denoised images')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DnCNN(depth=args.depth).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    psnrs = []
    for path in sorted(glob.glob(os.path.join(args.test_dir, '*.png'))):
        clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        np.random.seed(args.seed)
        noisy = clean + np.random.normal(0, args.sigma / 255.0, clean.shape).astype(np.float32)

        x = torch.from_numpy(noisy).view(1, 1, *clean.shape).to(device)
        denoised = (x - model(x)).clamp(0.0, 1.0).cpu().numpy().squeeze()

        psnr = calculate_psnr(denoised, clean)
        psnrs.append(psnr)
        name = os.path.basename(path)
        print(f'{name:15s} {psnr:.2f} dB')

        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, name),
                        (denoised * 255.0).round().astype(np.uint8))

    print(f'\naverage PSNR over {len(psnrs)} images (sigma={args.sigma}): {np.mean(psnrs):.2f} dB')


if __name__ == '__main__':
    main()
