import torch
import torch.nn as nn

class StripConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(StripConv, self).__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, 1
        else:
            kh, kw = kernel_size
        
        padding = ((kh - 1) // 2, (kw - 1) // 2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kh, kw), padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RAM(nn.Module):
    """
    Road Augmentation Module (RAM)
    Captures semantic shape information of roads using strip convolutions.
    Placed behind the fourth stage of the encoder.
    """
    def __init__(self, in_channels):
        super(RAM, self).__init__()
        inter_channels = in_channels // 4
        
        self.branch1 = StripConv(in_channels, inter_channels, (1, 7))
        self.branch2 = StripConv(in_channels, inter_channels, (7, 1))
        self.branch3 = StripConv(in_channels, inter_channels, (1, 11))
        self.branch4 = StripConv(in_channels, inter_channels, (11, 1))
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = x
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.conv_out(out)
        
        return out + identity
