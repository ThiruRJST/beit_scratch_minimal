import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class DVAETrainEngine(LightningModule):
    def __init__(self, model, criterion, lr):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters("criterion", "lr", ignore=["model"])
    
    def training_step(self, batch, batch_idx):
        x = batch
        xhat, logits = self.model(x)
        loss = self.criterion(x, xhat, logits)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        xhat, logits = self.model(x)
        loss = self.criterion(x, xhat, logits)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        return optimizer
    



