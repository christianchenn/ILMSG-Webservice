import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F

from src.models.modules.GaussianNoiseLayer import GaussianNoiseLayer


class AudioV2AEV2(pl.LightningModule):
    def __init__(self, learning_rate, run_name):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_name = "ilmsg-audioaev2"
        self.run_name = run_name
        self.learning_rate = learning_rate
        # nn.BatchNorm1d(num_features=512),
        self.encoder_lstm = nn.LSTM(128, 128, 1, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=128, out_features=512
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=512, out_features=256
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=256, out_features=128
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=128, out_features=64
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=64, out_features=32
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=32, out_features=32
            ),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=32,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=32,
                out_features=64,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=64,
                out_features=128,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=128,
                out_features=128,
            ),
        )
        self.decoder_lstm = nn.LSTM(128, 128, 1, batch_first=True)

    def forward(self, x):
        x, _ = self.encoder_lstm(x)
        x = self.encoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        out, (hn, cn) = self.encoder_lstm(x)
        z = self.encoder(out)
        # z = self.gaussian_layer(z)
        x_hat = self.decoder(z)
        out, (hn, cn) = self.decoder_lstm(x_hat, (hn, cn))
        loss = F.mse_loss(out, x)
        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        out, (hn, cn) = self.encoder_lstm(x)
        z = self.encoder(out)
        # z = self.gaussian_layer(z)
        x_hat = self.decoder(z)
        out, (hn, cn) = self.decoder_lstm(x_hat, (hn, cn))
        val_loss = F.mse_loss(out, x)
        self.log("val_loss", val_loss, logger=True, on_epoch=True, sync_dist=True)
        return x_hat

    def test_step(self, test_batch, batch_idx):
        x = test_batch
        out, (hn, cn) = self.encoder_lstm(x)
        z = self.encoder(out)
        # z = self.gaussian_layer(z)
        x_hat = self.decoder(z)
        out, (hn, cn) = self.decoder_lstm(x_hat, (hn, cn))
        test_loss = F.mse_loss(out, x)
        self.log("test_loss", test_loss, logger=True, on_epoch=True, sync_dist=True)
        return x_hat
