from torch import nn
import torch

class DiceCoefficient(nn.Module):
    def __init__(self, smooth = 1., eps=1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        intersection = (y_hat * y).sum()
        return (2. * intersection + self.smooth) / (y_hat.sum() + y.sum() + self.smooth + self.eps)

class IoUScore(nn.Module):
    def __init__(self, smooth = 1., eps=1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        intersection = (y_hat * y).sum()
        return (intersection + self.smooth) / (y_hat.sum() + y.sum() - intersection + self.smooth + self.eps)

