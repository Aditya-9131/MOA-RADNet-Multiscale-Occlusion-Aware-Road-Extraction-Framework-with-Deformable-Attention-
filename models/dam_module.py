import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DAM(nn.Module):
    """
    Deformable Attention Module (DAM)
    Combines deformable convolution and spatial attention to handle irregular road shapes
    and model long-range dependencies.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DAM, self).__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        
        # Offset generator for Deformable Conv
        # Each pixel has 2 offsets (dx, dy) for each kernel weight
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, padding=padding)
        
        self.deform_conv = DeformConv2d(in_channels, out_channels, 
                                        kernel_size=kernel_size, padding=padding, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # x is usually the combined features from encoder and decoder
        offsets = self.offset_conv(x)
        out = self.deform_conv(x, offsets)
        out = self.bn(out)
        out = self.relu(out)
        
        # Apply spatial attention
        att = self.spatial_attention(out)
        return out * att
