import math
from torch import nn

# Based on the explanation given in the paper,
# Number of layers (l) = 3
# kernel, i/p (f_i, n_i) where (5,64) -> (3, 32) -> 3
# GELU paper https://arxiv.org/pdf/1606.08415v3.pdf
# GELU incorporates regularisation (dropout) inherently. It demonstrates improvements in Computer Vision tasks.

class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            # used GeLU as activation function PSNR -> 32.99
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.GELU(),
        )
        # pixel shuffle is basically up-sampling the data from LR -> HR
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x


