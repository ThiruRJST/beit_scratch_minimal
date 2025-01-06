import mlflow.pytorch
import pandas as pd
import torch
import yaml

from data_loader import DVAEDataset
from dvae_loss import DVAEELBOLoss
from models.model_builder import build_dvae_model
from models.blocks import EncoderBlock, DecoderBlock
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training_engine import DVAETrainEngine



config = yaml.safe_load(open("params.yaml", "r"))
encoder_cfgs = config["DVAEEncoder"]
codebook_cfgs = config["DVAECodebook"]
dvaeloss_cfgs = config["DVAELoss"]

if __name__ == "__main__":

    mlflow.pytorch.autolog(
        log_models=True,
    )

    encoder = EncoderBlock(
        in_channels=3,
        out_channels=encoder_cfgs["codebook_dim"],
        n_layers=encoder_cfgs["num_layers"],
        n_res_blocks=encoder_cfgs["num_resnet_blocks"],
    )

    decoder = DecoderBlock(
        in_channels=encoder_cfgs["codebook_dim"],
        out_channels=3,
        n_layers=encoder_cfgs["num_layers"],
        n_res_blocks=encoder_cfgs["num_resnet_blocks"],
    )

    model = build_dvae_model(
        encoder=encoder,
        decoder=decoder,
        temperature=codebook_cfgs["gumbel_temperature"],
        codebook_size=codebook_cfgs["codebook_size"],
        codebook_dim=encoder_cfgs["codebook_dim"],
    )

    criterion = DVAEELBOLoss(
        kl_div_weight=dvaeloss_cfgs["kl_weight"],
        codebook_size=codebook_cfgs["codebook_size"],
    )


    train_csv = pd.read_csv("../data/imagewoof2/noisy_imagewoof_train.csv")
    valid_csv = pd.read_csv("../data/imagewoof2/noisy_imagewoof_valid.csv")

    train_paths = train_csv["path"].values
    valid_paths = valid_csv["path"].values


    train_dataset = DVAEDataset(train_paths)
    valid_dataset = DVAEDataset(valid_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=8,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=8,
    )

    train_engine = DVAETrainEngine(
        model=model,
        criterion=criterion,
        lr=config["train"]["lr"]
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="checkpoints",
        filename="dvae-{epoch:02d}-{val_loss:.2f}",
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=500,
        accelerator="gpu",
        callbacks=[checkpoint_cb, early_stopping_cb],
        precision=16
    )

    with mlflow.start_run(run_name="DVAE_Baseline"):
        trainer.fit(train_engine, train_loader, valid_loader)