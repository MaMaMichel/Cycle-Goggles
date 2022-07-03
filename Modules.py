import torch
from torch import nn,  exp, randn_like
from torch.nn import functional as F

class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1):
        super().__init__()
        self.subPixel = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel, stride, padding),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.subPixel(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, in_dim, kernel=3, stride=1, padding=1):
        super().__init__()
        self.resLayer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel, stride, padding),
            nn.BatchNorm2d(in_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        residual = x
        x = self.resLayer(x)
        x = self.resLayer(x)
        x = torch.add(residual, x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=1, padding=1):
        super().__init__()
        self.upLayers = nn.ModuleList([])
        for i in range(depth):
            self.upLayers.append(ResidualConnection(in_dim, kernel, stride=1, padding=padding))
        self.up = UpSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.upLayers:
            x = layer(x)
        x = self.up(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, kernel=3, stride=2, padding=1):
        super().__init__()
        self.resLayers = nn.ModuleList([])
        for i in range(depth):
            self.resLayers.append(ResidualConnection(in_dim, kernel, stride=1, padding=padding))
        self.down = DownSample(in_dim, out_dim, kernel, stride, padding)

    def forward(self, x):
        for layer in self.resLayers:
            x = layer(x)
        x = self.down(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, layer_dims=[3, 8, 16, 32, 64, 128, 256], input_size=128, output_size=1, layer_depth=2):
        super().__init__()

        self.Blocks = nn.ModuleList([])

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.Blocks.append(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))
        self.out_dim = layer_dims[-1]
        self.out_size = input_size // (2 ** (len(layer_dims) - 1))
        self.out = nn.Linear(self.out_dim * self.out_size * self.out_size, output_size)

    def forward(self, x):
        for block in self.Blocks:
            x = block(x)
        output = self.out(x.reshape(-1, self.out_dim * self.out_size * self.out_size))
        return output


class Generator(nn.Module):
    def __init__(self, layer_dims=[3, 8, 16, 32, 64, 128, 256], input_size=128, bottle_neck_size=16, latent_size=16, layer_depth=2):
        super().__init__()

        # Downsample
        self.layer_dims = layer_dims

        self.latent_size = latent_size
        self.bottle_neck_size = bottle_neck_size

        self.DownBlocks = nn.ModuleList([])

        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.DownBlocks.append(ResBlock(in_dim, out_dim, layer_depth, kernel=3, stride=2, padding=1))
        self.out_dim = self.layer_dims[-1]
        self.linear_size = input_size // (2 ** (len(self.layer_dims) - 1))
        self.squeeze = nn.Linear(self.out_dim * self.linear_size * self.linear_size, self.bottle_neck_size)

        # Upsample

        self.layer_dims.reverse()

        self.un_squeeze = nn.Linear(self.bottle_neck_size+self.latent_size,
                                    self.out_dim*self.linear_size*self.linear_size)

        self.UpBlocks = nn.ModuleList([])

        self.out = nn.Sigmoid()

        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.UpBlocks.append(UpBlock(in_dim * 2, out_dim, layer_depth, kernel=3, stride=1, padding=1))

    def forward(self, x, latent):
        skip_list = []
        for block in self.DownBlocks:
            x = block(x)
            skip_list.append(x)

        compressed_input = self.squeeze(x.reshape(-1, self.out_dim * self.linear_size * self.linear_size))

        x = self.un_squeeze(torch.cat(compressed_input, latent), 1)
        x = x.reshape(-1, self.in_dim, self.in_size, self.in_size)
        for block in self.UpBlocks:
            x = block(torch.cat(x, skip_list.pop()), 1)
        x = self.out(x)

        return x




