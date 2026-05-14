import torch
import torch.nn as nn
import torch.nn.functional as F

class ConnectivityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kernel', kernel)

    def forward(self, pred, target):
        laplacian = F.conv2d(pred, self.kernel, padding=1)
        return torch.mean(torch.abs(laplacian * target))

class HybridLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=0.5, w_conn=0.2):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_conn = w_conn
        self.conn_loss = ConnectivityLoss()

    def forward(self, pred, target):
        # Weighted BCE to favor road pixels (roads are sparse)
        weight = torch.tensor([5.0]).to(pred.device) # Penalize missing roads 5x more
        bce = F.binary_cross_entropy(pred, target, weight=weight)
        
        # Dice
        inter = (pred * target).sum()
        dice = 1 - (2. * inter + 1e-7) / (pred.sum() + target.sum() + 1e-7)
        
        # Connectivity
        conn = self.conn_loss(pred, target)
        
        return self.w_bce * bce + self.w_dice * dice + self.w_conn * conn
