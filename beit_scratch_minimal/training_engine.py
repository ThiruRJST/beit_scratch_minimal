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
        codebook_enc, logits = self.model.encode(x)
        decoded = self.model.decode(codebook_enc.permute(0, 3, 1, 2))

        total_loss, mse_loss, elbo_loss = self.criterion(x, decoded, logits)

        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_elbo_loss", elbo_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": total_loss, "total_train_loss": total_loss, "mse_train_loss": mse_loss, "elbo_train_loss": elbo_loss}
    
    def validation_step(self, batch, batch_idx):
        x = batch
        codebook_enc, logits = self.model.encode(x)
        decoded = self.model.decoder(codebook_enc.permute(0, 3, 1, 2))
        total_loss, mse_loss, elbo_loss = self.criterion(x, decoded, logits)

        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_elbo_loss", elbo_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"total_val_loss": total_loss, "mse_val_loss": mse_loss, "elbo_val_loss": elbo_loss}
    
    def configure_optimizers(self):
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        codebook_params = list(self.model.codebook.parameters())

        params_group = [
            {"params": encoder_params, "lr": self.hparams["lr"] * 0.1},
            {"params": decoder_params, "lr": self.hparams["lr"]},
            {"params": codebook_params, "lr": self.hparams["lr"] * 0.1},
        ]

        optimizer = optim.Adam(params_group, lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return [optimizer], [scheduler]
    



