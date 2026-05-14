import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import local modules
try:
    from .ram_module import RAM
    from .dam_module import DAM
except ImportError:
    from ram_module import RAM
    from dam_module import DAM

class DecoderBlock(nn.Module):
    def __init__(self, skip_channels, prev_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Reduce channels before DAM to save memory in deformable conv buffers
        self.reduction = nn.Sequential(
            nn.Conv2d(skip_channels + prev_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dam = DAM(out_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        feat = torch.cat([x, skip], dim=1)
        feat = self.reduction(feat) # Reduce channels before DAM
        out = self.dam(feat)
        out = self.conv(out)
        return out

class RADANet(nn.Module):
    """
    Road-Augmented Deformable Attention Network (RADANet)
    Backbone: ResNet50
    Modules: Road Augmentation Module (RAM), Deformable Attention Module (DAM)
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(RADANet, self).__init__()
        
        # Encoder: ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Initial layers
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # 64 ch, 1/2 size
        
        self.maxpool = resnet.maxpool # 1/4 size
        
        # ResNet layers
        self.layer1 = resnet.layer1 # 256 ch, 1/4 size
        self.layer2 = resnet.layer2 # 512 ch, 1/8 size
        self.layer3 = resnet.layer3 # 1024 ch, 1/16 size
        self.layer4 = resnet.layer4 # 2048 ch, 1/32 size
        
        # Road Augmentation Module (RAM) after layer4
        self.ram = RAM(2048)
        
        # Decoder Stages
        # D4: skips layer3 (1024), input from RAM (2048)
        self.decoder4 = DecoderBlock(1024, 2048, 512)
        
        # D3: skips layer2 (512), input from D4 (512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        
        # D2: skips layer1 (256), input from D3 (256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        
        # D1: skips encoder0 (64), input from D2 (128)
        self.decoder1 = DecoderBlock(64, 128, 64)
        
        # Final Prediction Layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)         # 1/2
        e1 = self.layer1(self.maxpool(e0)) # 1/4
        e2 = self.layer2(e1)               # 1/8
        e3 = self.layer3(e2)               # 1/16
        e4 = self.layer4(e3)               # 1/32
        
        # RAM
        e4_ram = self.ram(e4)
        
        # Decoder
        d4 = self.decoder4(e4_ram, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        
        # Output upsampling to original size
        out = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        
        return out

if __name__ == "__main__":
    # Test the model
    model = RADANet(num_classes=1)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
