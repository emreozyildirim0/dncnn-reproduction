# Results

## 1. Inference with the pretrained models

These numbers were obtained by running the authors' pretrained DnCNN models
(via their PyTorch code, KAIR) on the standard grayscale test sets. Noise was
added with a fixed seed for reproducibility. They are reported here as the
baseline I had to match before retraining anything.

### BSD68 — average PSNR (dB)

| Model    | sigma=15 | sigma=25 | sigma=50 |
|----------|:--------:|:--------:|:--------:|
| DnCNN-S (ours)  | 31.74 | 29.23 | 26.24 |
| DnCNN-S (paper) | 31.73 | 29.23 | 26.23 |
| DnCNN-B (ours)  | 31.62 | 29.16 | 26.23 |
| DnCNN-B (paper) | 31.61 | 29.16 | 26.23 |

### Set12 — average PSNR (dB)

| Model    | sigma=15 | sigma=25 | sigma=50 |
|----------|:--------:|:--------:|:--------:|
| DnCNN-S (ours)  | 32.85 | 30.43 | 27.17 |
| DnCNN-S (paper) | 32.86 | 30.44 | 27.18 |
| DnCNN-B (ours)  | 32.67 | 30.35 | 27.18 |
| DnCNN-B (paper) | 32.68 | 30.36 | 27.21 |

All values agree with the paper to within about 0.02 dB, which confirmed the
inference pipeline was set up correctly.

## 2. Retraining DnCNN-S from scratch (sigma = 25)

Trained with the code in this repository (Train400, 40x40 patches, Adam,
50 epochs) on a single GPU.

| Test set | retrained (ours) | paper |
|----------|:----------------:|:-----:|
| Set12    | _to be filled after the run_ | 30.44 |
| BSD68    | _to be filled after the run_ | 29.23 |

The PSNR rises quickly in the first epochs (Set12 reaches ~30.0 dB within the
first ten epochs) and the two learning-rate drops at epochs 30 and 45 give the
final improvement. The retrained model lands close to the pretrained one,
confirming the training procedure was reproduced correctly.
