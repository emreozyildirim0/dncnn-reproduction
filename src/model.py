import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN for Gaussian image denoising.

    The network is trained with residual learning: given a noisy image
    y = x + v it predicts the noise residual v, and the clean image is
    recovered afterwards as x = y - v.

    Architecture follows the paper:
      - first layer:  Conv + ReLU
      - middle layers: Conv + BatchNorm + ReLU   (depth - 2 of them)
      - last layer:   Conv
    All convolutions are 3x3 with 64 feature maps and zero padding so the
    spatial size is preserved.
    """

    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=1e-4, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)   # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
