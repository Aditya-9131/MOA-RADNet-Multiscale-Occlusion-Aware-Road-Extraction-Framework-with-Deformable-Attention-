import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import DeformConv2d

# =====================================================================
# 1. MSI: Multiscale Interaction Module (MSGCNet inspired)
# =====================================================================

class MSIModule(nn.Module):
    def __init__(self, in_channels_list, out_channels=512):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        self.query_conv = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.key_conv   = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.value_conv = nn.Conv2d(out_channels, out_channels, 1)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        target_size = projected[0].shape[2:]
        projected = [F.interpolate(p, size=target_size, mode='bilinear', align_corners=True) for p in projected]
        
        fused_list = []
        B, C, H, W = projected[0].shape
        for p in projected:
            q = self.query_conv(p).view(B, -1, H*W)
            k = self.key_conv(p).view(B, -1, H*W)
            v = self.value_conv(p).view(B, -1, H*W)
            
            attn = torch.bmm(q.transpose(1, 2), k)
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(v, attn.transpose(1, 2))
            fused_list.append(out.view(B, C, H, W))
            
        out = torch.cat(fused_list, dim=1)
        return self.fuse(out)

# =====================================================================
# 2. RAM: Road-Aware Module (RADANet inspired)
# =====================================================================

class DiagonalStripConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=11, direction='left'):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding, bias=False)
        mask = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            if direction == 'left': mask[i, i] = 1
            else: mask[i, kernel_size - 1 - i] = 1
        self.register_buffer('mask', mask.view(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.conv.weight * self.mask, padding=self.padding)

class RAMModule(nn.Module):
    def __init__(self, in_channels, k=11):
        super().__init__()
        mid = in_channels // 4
        self.horiz = nn.Conv2d(in_channels, mid, (1, k), padding=(0, k//2))
        self.vert  = nn.Conv2d(in_channels, mid, (k, 1), padding=(k//2, 0))
        self.diag1 = DiagonalStripConv(in_channels, mid, k, 'left')
        self.diag2 = DiagonalStripConv(in_channels, mid, k, 'right')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        feat = torch.cat([self.horiz(x), self.vert(x), self.diag1(x), self.diag2(x)], dim=1)
        return self.fuse(feat) + x

# =====================================================================
# 3. DAM: Deformable Attention Module (RADANet inspired)
# =====================================================================

class DAMModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=kernel_size//2)
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels//16, 1), nn.ReLU(), nn.Conv2d(in_channels//16, in_channels, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        offsets = self.offset_conv(x)
        out = self.bn(self.deform_conv(x, offsets))
        out = out * self.ca(out)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        out = out * self.sa(torch.cat([avg_out, max_out], dim=1))
        return out

# =====================================================================
# 4. OADecoder: Occlusion-Aware Decoder (OARENet inspired)
# =====================================================================

class OADecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn_map = nn.Sequential(nn.Conv2d(out_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > 0.05).float()
            x = x * mask
        feat = self.dilated(x)
        weights = self.attn_map(feat)
        return feat * weights

# =====================================================================
# FINAL HYBRID MODEL
# =====================================================================

class HybridRADANet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.ModuleDict({
            'stem': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            'c1': resnet.layer1, 'c2': resnet.layer2, 'c3': resnet.layer3, 'c4': resnet.layer4
        })
        self.msi = MSIModule([512, 1024, 2048], 512)
        self.ram = RAMModule(512)
        self.dam = DAMModule(512)
        self.oa_decoder = OADecoder(512, 256)
        self.head = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, num_classes, 1), nn.Sigmoid())

    def forward(self, x):
        s = self.backbone['stem'](x)
        c1 = self.backbone['c1'](s)
        c2 = self.backbone['c2'](c1)
        c3 = self.backbone['c3'](c2)
        c4 = self.backbone['c4'](c3)
        
        out = self.msi([c2, c3, c4])
        out = self.ram(out)
        out = self.dam(out)
        out = self.oa_decoder(out)
        
        out = self.head(out)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
