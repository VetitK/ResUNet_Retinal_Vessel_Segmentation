import pytorch_lightning as pl
from torch.optim import Adam
from model.model import ResUNet, ResUnetVariant2
from losses import DiceLoss, DiceBCELoss
from metrics import DiceCoefficient, IoUScore
import torch
from typing import Literal
class RetinaVesselSegmentation(pl.LightningModule):
    def __init__(self, data_dir: str = 'data', lr: float = 1e-7, loss_type: Literal['BCE', 'Dice', 'DiceBCE'] = 'BCE', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.lr = lr
        self.model = ResUnetVariant2(channel=3)

        if loss_type == 'BCE':
            self.loss = torch.nn.BCELoss() 
        elif loss_type == 'Dice':
            self.loss = DiceLoss()
        elif loss_type == 'DiceBCE':
            self.loss = DiceBCELoss()

        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        return opt
    
    def training_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        
        dice_coeff = DiceCoefficient()(y_hat, y)
        iou_score = IoUScore()(y_hat, y)
        
        self.log('train_loss', loss)
        self.log('train_dice_coeff', dice_coeff)
        self.log('train_iou_score', iou_score)
        return loss
    
    def validation_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        dice_coeff = DiceCoefficient()(y_hat, y)
        iou_score = IoUScore()(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_dice_coeff', dice_coeff)
        self.log('val_iou_score', iou_score)

        return loss
    
    def test_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        dice_coeff = DiceCoefficient()(y_hat, y)
        iou_score = IoUScore()(y_hat, y)
        
        self.log('test_loss', loss)
        self.log('test_dice_coeff', dice_coeff)
        self.log('test_iou_score', iou_score)
        return loss