from torch import nn
import torch
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, eps=1e-7, stride=1, depth=1, is_decoder=False):
        super(ResNetBlock, self).__init__()
        
        self.depth = depth
        self.is_decoder = is_decoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        residual = self.conv3(x)
        residual = self.bn3(residual)
        # print('residual', residual.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print('conv1', x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print('conv2', x.shape)
        x = x + residual
        if not self.is_decoder and self.depth > 1:
            x = self.pool(x)
        # print('x', x.shape)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.depth = depth
        
    def forward(self, x):
        x = ResNetBlock(self.in_channels, self.out_channels, depth=self.depth, stride=self.stride)(x)
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, in_channel, out_channel, depth, eps=1e-7, stride=1, ) -> None:
        super(Decoder, self).__init__()
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = ResNetBlock(self.in_channel, self.out_channel, depth=self.depth, stride=1, is_decoder=True)(x)

        return x

    
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)