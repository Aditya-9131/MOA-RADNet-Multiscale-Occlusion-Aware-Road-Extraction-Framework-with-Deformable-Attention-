import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(feat))

class DAM(nn.Module):
    """
    Deformable Attention Module (DAM)
    Combines Deformable Convolution and Spatial Attention.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DAM, self).__init__()
        padding = kernel_size // 2
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, 
                                        kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        offsets = self.offset_conv(x)
        out = self.deform_conv(x, offsets)
        out = self.bn(out)
        out = self.relu(out)
        return out * self.spatial_att(out)
