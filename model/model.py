import torch.nn as nn
import torch
from torch.functional import Tensor
from model.block import Encoder, Decoder, ResNetBlock, ResidualConv, Upsample
class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, eps=1e-7, stride=1, is_activation_layer: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.eps = eps
        self.encoder1 = ResNetBlock(in_channels, 64, depth=1, stride=stride)
        self.encoder2 = ResNetBlock(64, 128, depth=2, stride=stride)
        self.encoder3 = ResNetBlock(128, 256, depth=3, stride=stride)
        self.encoder4 = ResNetBlock(256, 512, depth=4, stride=stride)
        self.decoder1 = Decoder(512, 256, depth=3)
        self.decoder2 = Decoder(256, 128, depth=2)
        self.decoder3 = Decoder(128, 64, depth=1)
        self.fc = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)
        if is_activation_layer:
            self.fc = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=1, stride=1), nn.Sigmoid())
    
    def forward(self, x):
        x1_down = self.encoder1(x)
        x2_down = self.encoder2(x1_down)
        x3_down = self.encoder3(x2_down)
        x4 = self.encoder4(x3_down)
        x3_up = self.decoder1(x4, x3_down)
        x2_up = self.decoder2(x3_up, x2_down)
        x1_up = self.decoder3(x2_up, x1_down)
        x = self.fc(x1_up)

        return x

class ResUnetVariant2(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnetVariant2, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output