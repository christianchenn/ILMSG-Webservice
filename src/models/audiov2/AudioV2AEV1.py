import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F

from src.models.modules.GaussianNoiseLayer import GaussianNoiseLayer


class AudioV2AEV1(pl.LightningModule):
    def __init__(self, learning_rate, run_name):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_name = "ilmsg-audioaev2"
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=128, out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512, out_features=128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128, out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64, out_features=32,
            ),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.gaussian_layer = GaussianNoiseLayer(0.05)
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=64,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=128,
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        z = self.encoder(train_batch)
        z = self.gaussian_layer(z)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z = self.encoder(val_batch)
        z = self.gaussian_layer(z)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss, logger=True, on_epoch=True, sync_dist=True)
        return x_hat

    def test_step(self, test_batch, batch_idx):
        x = test_batch
        z = self.encoder(test_batch)
        z = self.gaussian_layer(z)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss, logger=True, on_epoch=True, sync_dist=True)
        return x_hat
