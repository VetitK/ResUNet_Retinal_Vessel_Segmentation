from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        intersection = (y_hat * y).sum()
        return 1 - ((2. * intersection + self.smooth) / (y_hat.sum() + y.sum() + self.smooth + self.eps))

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7, weight: float = 0.5) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.weight = weight
        self.dice_loss = DiceLoss(smooth=self.smooth, eps=self.eps)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.weight * self.dice_loss(y_hat, y) + (1 - self.weight) * self.bce_loss(y_hat, y)

