import torch
import wandb
import pytorch_lightning as pl
from datasets import HMC
from models import Lit3D
import argparse

parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--net', type=str, default='r2p1d')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--subset', type=int, default=1)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--scratch', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=True)
args = parser.parse_args()
print(args)

gpu = args.gpu
lr = args.lr
net = args.net
subset = args.subset
pretrained = args.pretrained
tbs = args.bs
epochs = args.epochs
model_path = args.model_path
seg = None

accs = []
f1s = []
specs = []
precs = []
recs = []

for fold in range(1,6):
    
    train_dataset = HMC(split='train', fold=fold, seg=seg, sub=subset)
    test_dataset = HMC(split='test', fold=fold, seg=seg, sub=subset)

    print(len(train_dataset), len(test_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=tbs, shuffle=True,
        num_workers=24, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2, shuffle=False,
        num_workers=24, pin_memory=True)

    pl.utilities.seed.seed_everything(3407)
    logger = pl.loggers.WandbLogger(
        project='HMC', name=f'a4c_{lr}_net_{net}_subset_{subset}_bs_{tbs}_network_{net}_fold_{fold}')
    trainer = pl.Trainer(max_epochs=epochs, gpus=[gpu], logger=logger, check_val_every_n_epoch=epochs)
    model = Lit3D(mlr=lr, network=net, pretrained=pretrained, model_path=model_path)
    trainer.fit(model, train_dataloader, test_dataloader)
    
    metrics = trainer.logged_metrics
    accs.append(metrics['val_acc'])
    f1s.append(metrics['val_f1'])
    specs.append(metrics['val_spec'])
    precs.append(metrics['val_prec'])
    recs.append(metrics['val_rec'])
    wandb.finish()

print('Average metrics over the 5 folds:\n')
print('Accuracy:', sum(accs) / 5)
print('F1:', sum(f1s) / 5)
print('Specificity:', sum(specs) / 5)
print('Precision:', sum(precs) / 5)
print('Sensitivity:', sum(recs) / 5)
