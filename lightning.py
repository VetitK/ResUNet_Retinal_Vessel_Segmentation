import pytorch_lightning as pl
from torch.optim import Adam
from model.model import ResUNet, ResUnetVariant2
import torch
class RetinaVesselSegmentation(pl.LightningModule):
    def __init__(self, data_dir: str = 'data', lr: float = 1e-7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.lr = lr
        self.model = ResUnetVariant2(channel=3)
        self.loss = torch.nn.BCELoss()
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
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss