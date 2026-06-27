# DnCNN — Residual Learning of Deep CNN for Image Denoising

A reimplementation and reproduction of the Gaussian denoising experiments from

> K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang,
> *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*,
> IEEE Transactions on Image Processing, 26(7), 2017.

The project has two parts. First I ran the original pretrained models at
inference to make sure I could reproduce the PSNR values reported in the paper.
Then I reimplemented the DnCNN-S model and retrained it from scratch to go
through the training procedure myself and understand the details that the paper
leaves implicit.

## The method in short

DnCNN does not output the clean image directly. With a noisy observation
`y = x + v` it is trained to predict the noise `v` (this is the *residual
learning* idea), and the clean image is recovered as `x = y - v`. Predicting the
residual is easier here because the residual mapping is close to what the
network would otherwise have to subtract, and it combines well with batch
normalization to give fast and stable training.

The network is a plain feed-forward CNN:

- depth 17 for a known noise level (DnCNN-S), depth 20 for the blind model (DnCNN-B);
- 3×3 convolutions, 64 feature maps, padding so the spatial size never changes;
- `Conv+ReLU` for the first layer, `Conv+BN+ReLU` for the middle layers, a single
  `Conv` for the last layer;
- He (Kaiming) initialization;
- loss is the mean squared error between the predicted and the true residual,
  averaged as in Eq. (1) of the paper.

## Repository layout

```
src/
  model.py     DnCNN network
  dataset.py   on-the-fly patch extraction, augmentation and noise
  train.py     training loop (command line)
  test.py      evaluation / denoising of a test set
  utils.py     PSNR and evaluation helpers
notebooks/
  DnCNN_train_kaggle.ipynb   same training, ready to run on Kaggle/Colab with a GPU
data/
  README.md    where to get Train400, Set12 and BSD68
```

## Setup

```
pip install -r requirements.txt
```

`numpy` is pinned to `1.26.4`: the prebuilt torch wheels are compiled against the
numpy 1.x ABI and importing them with numpy 2.x fails.

## Data

Download Train400, Set12 and BSD68 from the original repository
(https://github.com/cszn/DnCNN) and place them under `data/` as described in
`data/README.md`.

## Training

```
python src/train.py --train_dir data/Train400 --val_dir data/Set12 --sigma 25 --epochs 50
```

This trains DnCNN-S for σ = 25 with Adam, evaluating on Set12 after every epoch
and keeping the best checkpoint. Training on CPU is slow; a GPU is strongly
recommended. The notebook in `notebooks/` is the easiest way to run it on a free
GPU (Kaggle or Colab) — it is the same code with the data paths set up for that
environment.

## Testing

```
python src/test.py --checkpoint dncnn_s25.pth --test_dir data/Set12 --sigma 25
python src/test.py --checkpoint dncnn_s25.pth --test_dir data/BSD68 --sigma 25 --save_dir outputs/bsd68
```

`--save_dir` is optional and writes the denoised images.

## Inference baseline (pretrained models)

For the baseline I used the authors' own PyTorch code, KAIR
(https://github.com/cszn/KAIR), with the released pretrained weights, e.g.

```
python main_test_dncnn.py --model_name dncnn_25 --testset_name set12 --noise_level_img 25
```

The resulting PSNR values match the paper to within about 0.02 dB.

## Notes and difficulties

- **Optimizer.** The paper states SGD with an initial learning rate of 0.1. With
  this loss (summed over the batch, Eq. 1) my PyTorch reimplementation diverged
  to NaN within a few iterations at that learning rate. Switching to Adam
  (lr = 1e-3) — which is also what the commonly used reimplementations do — made
  training stable and converged to the same PSNR; the paper itself notes (Fig. 2)
  that SGD and Adam reach the same accuracy. This was the main practical issue.
- **Reproducible evaluation.** During training and testing the noise is generated
  with a fixed seed, otherwise the per-image PSNR fluctuates and runs are not
  comparable.
- **Patch sampling.** One "epoch" here means a fixed number of randomly cropped
  patches (1600 × batch size by default) rather than a single pass over the 400
  images, following the original training setup.

## References

1. K. Zhang et al., *Beyond a Gaussian Denoiser*, IEEE TIP 2017.
2. Original code and data: https://github.com/cszn/DnCNN
3. PyTorch toolbox used for the pretrained inference baseline: https://github.com/cszn/KAIR
