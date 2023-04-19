from torch import nn
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1,padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Block(in_channels,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x_skip = self.conv(x)
        x = self.pool(x_skip)
        return x,x_skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = Block(in_channels,out_channels)
        

    def forward(self, x,skip):
        x = self.up(x)
        x = torch.cat([x,skip],dim=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64,depth=4, lr=1e-3,optimizer=torch.optim.Adam):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input_layer = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)

        # down part
        in_feature = in_channels
        out_feature = features
        for _ in range(depth):
            self.downs.append(DownBlock(in_feature, out_feature))
            in_feature = out_feature
            out_feature = out_feature*2

        # middle
        self.bottleneck = Block(in_feature,out_feature)

        # up part
        for _ in range(depth):
            in_feature = out_feature 
            out_feature = in_feature//2
            self.ups.append(UpBlock(in_feature, out_feature))

        self.output_layer = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.input_layer(x)
        for down in self.downs:
            x,x_skip = down(x)
            skip_connections.append(x_skip)

        x = self.bottleneck(x)

        for up in self.ups:
            x = up(x,skip_connections.pop())

        x = self.output_layer(x)
        return x