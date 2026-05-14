import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .ram import RAM
from .dam import DAM

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dam = DAM(in_channels + skip_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        feat = torch.cat([x, skip], dim=1)
        out = self.dam(feat)
        return self.conv(out)

class RADANet(nn.Module):
    def __init__(self, num_classes=1):
        super(RADANet, self).__init__()
        
        # Encoder: Pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64, 1/2
        self.maxpool = resnet.maxpool # 1/4
        self.enc1 = resnet.layer1 # 256, 1/4
        self.enc2 = resnet.layer2 # 512, 1/8
        self.enc3 = resnet.layer3 # 1024, 1/16
        self.enc4 = resnet.layer4 # 2048, 1/32
        
        # RAM Module
        self.ram = RAM(2048)
        
        # Decoder
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)
        x1 = self.enc1(self.maxpool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        # RAM
        x4 = self.ram(x4)
        
        # Decoder
        y4 = self.dec4(x4, x3)
        y3 = self.dec3(y4, x2)
        y2 = self.dec2(y3, x1)
        y1 = self.dec1(y2, x0)
        
        # Final prediction
        out = F.interpolate(y1, size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.sigmoid(self.final_conv(out))
