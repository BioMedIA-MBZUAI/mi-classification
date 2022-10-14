import torch
import pytorch_lightning as pl
from dataset import HMC
from model import Lit3D
import argparse

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--net', type=str, default='r2p1d')
parser.set_defaults(pretrained=True)
args = parser.parse_args()
print(args)

gpu = args.gpu
lr = args.lr
net = args.net
tbs = args.bs
epochs = args.epochs

train_dataset = HMC(split='train')
val_dataset = HMC(split='test')

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=tbs, shuffle=True,
    num_workers=4, pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=2, shuffle=False,
    num_workers=4, pin_memory=True,drop_last=True)

pl.utilities.seed.seed_everything(3407)
logger = pl.loggers.WandbLogger(project='HMC', name='ef_pretraining')
checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath='../', filename='r2p_model')
trainer = pl.Trainer(max_epochs=epochs, callbacks=[
                     checkpoint], gpus=[gpu], logger=logger)
model = Lit3D(network=net, mlr=lr)
trainer.fit(model, train_dataloader, val_dataloader)
