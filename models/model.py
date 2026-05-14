import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import DeformConv2d

# ==========================================
# 2. Road Augmentation Module (RAM)
# ==========================================

class DiagonalStripConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, direction='left'):
        super().__init__()
        self.kernel_size = kernel_size
        self.direction = direction
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=self.padding, bias=False)
        
        mask = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            if direction == 'left': # Top-left to bottom-right
                mask[i, i] = 1
            else: # Top-right to bottom-left
                mask[i, kernel_size - 1 - i] = 1
        
        self.register_buffer('mask', mask.view(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        masked_weight = self.conv.weight * self.mask
        return F.conv2d(x, masked_weight, bias=None, padding=self.padding)

class RAM(nn.Module):
    """
    Road Augmentation Module (RAM) strictly following paper-inspired 4-direction logic.
    Horizontal, Vertical, Left Diagonal, Right Diagonal.
    """
    def __init__(self, in_channels, kernel_size=11):
        super(RAM, self).__init__()
        inter_channels = in_channels // 4
        
        # Horizontal (1xK)
        self.horiz = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Vertical (Kx1)
        self.vert = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Left Diagonal
        self.l_diag = nn.Sequential(
            DiagonalStripConv(in_channels, inter_channels, kernel_size, direction='left'),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Right Diagonal
        self.r_diag = nn.Sequential(
            DiagonalStripConv(in_channels, inter_channels, kernel_size, direction='right'),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = x
        
        h = self.horiz(x)
        v = self.vert(x)
        ld = self.l_diag(x)
        rd = self.r_diag(x)
        
        # Element-wise summation fuse as requested
        out = torch.cat([h, v, ld, rd], dim=1)
        out = self.conv_fuse(out)
        
        return out + identity

# ==========================================
# 3. Deformable Attention Module (DAM)
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

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
    Combines Deformable Conv + Channel Attention + Spatial Attention.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_deform=True):
        super(DAM, self).__init__()
        self.use_deform = use_deform
        padding = kernel_size // 2
        
        if self.use_deform:
            self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
            self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        else:
            # Fallback to standard Conv2d for CPU efficiency/stability if requested
            self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        if self.use_deform:
            offsets = self.offset_conv(x)
            out = self.deform_conv(x, offsets)
        else:
            out = self.deform_conv(x)
            
        out = self.bn(out)
        out = self.relu(out)
        
        # Dual Attention
        out = out * self.channel_att(out)
        out = out * self.spatial_att(out)
        return out

# ==========================================
# 1 & 4. RADANet (Encoder + Decoder)
# ==========================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.2, use_deform=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # DAM applied between encoder and decoder stages
        self.dam = DAM(in_channels + skip_channels, out_channels, use_deform=use_deform)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Ensure sizes match (in case of odd input dims)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1)
        x = self.dam(x)
        return self.conv(x)

class RADANet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, use_deform=True):
        super(RADANet, self).__init__()
        self.use_deform = use_deform
        
        # Encoder: Pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64 ch, 1/2
        self.maxpool = resnet.maxpool # 1/4
        self.enc1 = resnet.layer1 # 256 ch, 1/4
        self.enc2 = resnet.layer2 # 512 ch, 1/8
        self.enc3 = resnet.layer3 # 1024 ch, 1/16
        self.enc4 = resnet.layer4 # 2048 ch, 1/32
        
        # RAM after Stage 4
        self.ram = RAM(2048)
        
        # Decoder
        self.dec4 = DecoderBlock(2048, 1024, 512, dropout_rate, use_deform=use_deform)
        self.dec3 = DecoderBlock(512, 512, 256, dropout_rate, use_deform=use_deform)
        self.dec2 = DecoderBlock(256, 256, 128, dropout_rate, use_deform=use_deform)
        self.dec1 = DecoderBlock(128, 64, 64, dropout_rate, use_deform=use_deform)
        
        # Final Decoder stage for 1/1 resolution
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final Output Layer (1x1 conv + Sigmoid)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)             # 1/2
        x1 = self.enc1(self.maxpool(x0)) # 1/4
        x2 = self.enc2(x1)            # 1/8
        x3 = self.enc3(x2)            # 1/16
        x4 = self.enc4(x3)            # 1/32
        
        # RAM
        x4 = self.ram(x4)
        
        # Decoder
        y4 = self.dec4(x4, x3)        # 1/16
        y3 = self.dec3(y4, x2)        # 1/8
        y2 = self.dec2(y3, x1)        # 1/4
        y1 = self.dec1(y2, x0)        # 1/2
        
        # Final stage to 1/1
        y0 = self.dec0(y1)            # 1/1
        
        out = self.final_conv(y0)
        return self.sigmoid(out)

if __name__ == "__main__":
    model = RADANet()
    test_in = torch.randn(1, 3, 512, 512)
    test_out = model(test_in)
    print(f"RADANet Output Shape: {test_out.shape}")
