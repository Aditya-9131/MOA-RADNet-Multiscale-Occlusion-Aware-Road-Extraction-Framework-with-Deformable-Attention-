"""
RADANet Loss Functions
=====================
Composite loss: 0.3 * BCE + 0.5 * Dice + 0.2 * Focal
All losses expect sigmoid-activated predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance in road extraction."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Numerical stability clamp
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)

        # Binary Cross Entropy
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        
        # Focal calculation
        pt = target * pred + (1.0 - target) * (1.0 - pred)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce
        
        return focal.mean()

class ImprovedRADALoss(nn.Module):
    """
    Composite loss for RADANet road segmentation.
    Loss = 0.3 * BCE + 0.5 * Dice + 0.2 * Focal
    """
    def __init__(self, w_bce=0.3, w_dice=0.5, w_focal=0.2):
        super(ImprovedRADALoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred, target):
        # BCE
        bce = F.binary_cross_entropy(pred, target)
        
        # Dice
        dice = self.dice_loss(pred, target)
        
        # Focal
        focal = self.focal_loss(pred, target)
        
        return self.w_bce * bce + self.w_dice * dice + self.w_focal * focal
