import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r2plus1d_18, r3d_18
import pytorch_lightning as pl
import madgrad
from sklearn.metrics import mean_absolute_error


class Lit3D(pl.LightningModule):
    def __init__(self, network='r2p1d', mlr=1e-4):
        super().__init__()

        if network=='r2p1d':
            self.net = r2plus1d_18(pretrained=False, num_classes=1)
            self.net.stem[0] = nn.Conv3d(1, 45, kernel_size=(
                1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)

        else:
            self.net = r3d_18(pretrained=False, num_classes=1)
            self.net.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        self.net.fc.bias.data[0] = 55.6

        self.criterion = nn.MSELoss()
        self.learning_rate = mlr

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        if len(y_hat.shape) > 1:
            y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.float()
        y_pred = self(x)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.squeeze()
        loss = self.criterion(y_pred, y_true)
        mae = mean_absolute_error(y_pred.cpu(), y_true.cpu())
        self.log("val_mae", mae, prog_bar=True, logger=True)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = madgrad.MADGRAD(self.parameters(), lr=self.learning_rate)
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 20, eta_min=0, last_epoch=- 1, verbose=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }
