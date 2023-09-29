from typing import Dict
import pytorch_lightning as pl
import torch
from utilities import instantiate_from_config


class DiffusionDenoise(pl.LightningModule):
    def __init__(self, model_config: Dict, lr: float = 1e-5) -> None:
        super().__init__()
        self.lr = lr
        self.model = instantiate_from_config(model_config)

    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        err = torch.mean(torch.square(y - self(X)))
        self.log("train/loss", err.detach().mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return err

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer