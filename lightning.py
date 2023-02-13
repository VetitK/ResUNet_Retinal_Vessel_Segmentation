import pytorch_lightning as pl
from torch.optim import Adam
from model.model import ResUNet, ResUnetVariant2
from losses import DiceLoss, DiceBCELoss
from metrics import DiceCoefficient, IoUScore, Accuracy
import torch
from torchvision.transforms import Grayscale
from typing import Literal
import wandb
class RetinaVesselSegmentation(pl.LightningModule):
    def __init__(self,
                 lr: float = 1e-7,
                 loss_type: Literal['BCE', 'Dice', 'DiceBCE'] = 'BCE',
                 model: Literal['ResUNet', 'ResUnetVariant2'] = 'ResUnetVariant2',
                 img_size: int = 224) -> None:
        super().__init__()
        self.img_size = img_size
        self.lr = lr
        self.model = ResUnetVariant2(channel=3) if model == 'ResUnetVariant2' else ResUNet()

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
        acc = Accuracy()(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_dice_coeff', dice_coeff)
        self.log('train_iou_score', iou_score)
        self.log('train_acc', acc)
        return {'loss': loss, 'x': x, 'y_hat': y_hat, 'y': y}
    
    def training_epoch_end(self, outputs) -> None:
        x = outputs[0]['x']
        y_hat = outputs[0]['y_hat']
        y = outputs[0]['y']
        mask_img = wandb.Image(Grayscale()(x[0]), masks={
            "prediction": {
                "mask_data": torch.nn.Threshold(0.5, 0.0)(y_hat[0]).squeeze(0).detach().cpu().numpy(),
                "class_labels": {0: "background", 1: "vessel"}
            },
            "ground_truth": {
                "mask_data": y[0].squeeze(0).detach().cpu().numpy(),
                "class_labels": {0: "background", 1: "vessel"}
            }
        }, caption='Input Image Masks')
        self.logger.experiment.log({'images': [mask_img, wandb.Image(x[0], caption='raw'), wandb.Image(y_hat[0], caption='pred'), wandb.Image(y[0], caption='gt')]})
        
    def validation_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        dice_coeff = DiceCoefficient()(y_hat, y)
        iou_score = IoUScore()(y_hat, y)
        acc = Accuracy()(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_dice_coeff', dice_coeff)
        self.log('val_iou_score', iou_score)
        self.log('train_acc', acc)
        return loss
    
    def test_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        dice_coeff = DiceCoefficient()(y_hat, y)
        iou_score = IoUScore()(y_hat, y)
        acc = Accuracy()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_dice_coeff', dice_coeff)
        self.log('test_iou_score', iou_score)
        self.log('train_acc', acc)
        return loss