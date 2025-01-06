import albumentations as A
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder, EarlyStopping
from custom_data import ImageWoofDataset
from models import TimmNetWrapper
from torchmetrics import Accuracy, Precision, Recall
from torch.utils.data import DataLoader

class ImageWoofClassifier(pl.LightningModule):
    def __init__(self, model_name, num_heads, lr:float):
        super(ImageWoofClassifier, self).__init__()

        self.model = TimmNetWrapper(model_name, num_heads)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters("lr", "model_name", "num_heads")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return loss
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
    


if __name__ == '__main__':

    model_name = "efficientnet_b0"
    
    mlflow.pytorch.autolog(
        log_every_n_epoch=1,
        log_every_n_step=None,
        log_models=True,
        disable=False,
        log_datasets=True,
        registered_model_name=f"{model_name}_ImageWoof_masked"
    )
    
    with mlflow.start_run(run_name=f"{model_name}_ImageWoof"):
        model = ImageWoofClassifier(model_name, 24*24*3, 1e-3)
        
        train_data = pd.read_csv("../data/imagewoof2/noisy_imagewoof_train.csv")
        val_data = pd.read_csv("../data/imagewoof2/noisy_imagewoof_valid.csv")

        train_dataset = ImageWoofDataset(train_data['path'].values)
        val_dataset = ImageWoofDataset(val_data['path'].values)

        train_loader = DataLoader(train_dataset, batch_size=48, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=48, num_workers=4, shuffle=False)

        lrfinder_cb = LearningRateFinder()
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=4, mode='min')
        checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', save_weights_only=True)

        trainer = pl.Trainer(num_sanity_val_steps=2, accelerator="gpu", callbacks=[earlystop_cb, checkpoint_cb], max_epochs=500)

        trainer.fit(model, train_loader, val_loader)
        