import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DnCNN
from dataset import DenoisingDataset
from utils import evaluate


def parse_args():
    p = argparse.ArgumentParser(description='Train DnCNN-S for Gaussian denoising.')
    p.add_argument('--train_dir', required=True, help='folder with training images (e.g. Train400)')
    p.add_argument('--val_dir', default=None, help='folder used to track PSNR during training (e.g. Set12)')
    p.add_argument('--sigma', type=int, default=25, help='noise level the model is trained for')
    p.add_argument('--depth', type=int, default=17)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--patch_size', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--steps_per_epoch', type=int, default=1600, help='iterations per epoch')
    p.add_argument('--milestones', type=int, nargs='*', default=[30, 45],
                   help='epochs at which the learning rate is multiplied by 0.1')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out', default='dncnn_s25.pth', help='where to save the trained weights')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = DnCNN(depth=args.depth).to(device)
    print(f'parameters: {sum(p.numel() for p in model.parameters())}')

    # Loss as in Eq. (1) of the paper: squared error summed over the batch,
    # divided by twice the batch size.
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = [m for m in args.milestones if m < args.epochs]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    dataset = DenoisingDataset(args.train_dir, patch_size=args.patch_size, sigma=args.sigma,
                               length=args.steps_per_epoch * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        for noisy, noise in loader:
            noisy, noise = noisy.to(device), noise.to(device)
            optimizer.zero_grad()
            loss = criterion(model(noisy), noise) / (2 * noisy.size(0))
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        line = f'epoch {epoch:2d}/{args.epochs} | loss {running / len(loader):.4f}'
        if args.val_dir:
            psnr = evaluate(model, args.val_dir, sigma=args.sigma, device=device)
            line += f' | val PSNR {psnr:.2f} dB'
            if psnr > best:
                best = psnr
                torch.save(model.state_dict(), args.out)
        else:
            torch.save(model.state_dict(), args.out)
        line += f' | lr {optimizer.param_groups[0]["lr"]:.1e} | {time.time() - start:.0f}s'
        print(line)

    if args.val_dir:
        print(f'best val PSNR: {best:.2f} dB  (weights saved to {args.out})')
    else:
        print(f'weights saved to {args.out}')


if __name__ == '__main__':
    main()
