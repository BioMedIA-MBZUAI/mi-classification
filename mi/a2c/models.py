import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
import torchmetrics
import madgrad
from collections import OrderedDict


class Lit3D(pl.LightningModule):

    def __init__(self, mlr=1e-5, network='r2p1d', pretrained=True, model_path=None):
        super().__init__()

        if network == 'r2p1d':
            self.resnet = models.video.r2plus1d_18(
                pretrained=False, num_classes=1)
            self.resnet.stem[0] = nn.Conv3d(1, 45, kernel_size=(
                1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            print('#### NETWORK = r2p1d ####')

        else:
            self.resnet = models.video.r3d_18(pretrained=False, num_classes=1)
            self.resnet.stem[0] = nn.Conv3d(1, 64, kernel_size=(
                3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            print('#### NETWORK = r3d ####')

        new_dict = OrderedDict()

        if pretrained:
            if model_path:
                model_file = model_path
            else:
                model_file = '../r2p_model.ckpt' if network == 'r2p1d' else '../r3d_model.ckpt'
            state_dict = torch.load(model_file, map_location=torch.device('cuda'))['state_dict']
            print(f'### MODEL LOADED FROM: {model_file} ####')

            for key, value in state_dict.items():
                new_key = key.replace('net.', '')
                new_dict[new_key] = value

            
            self.resnet.load_state_dict(new_dict)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
            
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = mlr
        print(f'#### LEARNING RATE = {self.learning_rate} ####')

    def forward(self, x):
        x = self.resnet(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if y_hat.size(dim=0) > 1:
            y_hat = y_hat.squeeze()
        else:
            y_hat = y_hat[:, 0]
        loss = self.criterion(y_hat, y.float())
        acc = torchmetrics.Accuracy()(torch.round(torch.sigmoid(y_hat)).cpu(), y.cpu())
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x , y_true = batch
        y_pred = self(x)
        if y_pred.size(dim=0) > 1:
            y_pred = y_pred.squeeze()
        else:
            y_pred = y_pred[:,0]
        preds = torch.round(torch.sigmoid(y_pred))
        targets = y_true

        return preds, targets

    def validation_epoch_end(self, validation_step_outputs):
        preds = []
        targets = []

        for outs in validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        spec = torchmetrics.Specificity()(preds,targets)
        f1 = torchmetrics.F1()(preds,targets)
        prec = torchmetrics.Precision()(preds,targets)
        rec = torchmetrics.Recall()(preds,targets)
        acc = torchmetrics.Accuracy()(preds,targets)

        self.log("val_spec", spec, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)
        self.log("val_prec", prec, prog_bar=True, logger=True)
        self.log("val_rec", rec, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = madgrad.MADGRAD(self.parameters(), lr=self.learning_rate)
        return optimizer
