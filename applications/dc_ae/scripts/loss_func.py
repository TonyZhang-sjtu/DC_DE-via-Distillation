import torch.nn.functional as F
import torch.nn as nn

# Definition of PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def conv(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv(in_channels, 64, norm=False),
            conv(64, 128),
            conv(128, 256),
            conv(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)  # 输出 patch 判别图
        )

    def forward(self, x):
        return self.model(x)
    

