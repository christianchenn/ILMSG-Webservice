import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F


class AudioAEV7(pl.LightningModule):
    def __init__(self, learning_rate, run_name):
        super().__init__()
        self.save_hyperparameters()
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=24, kernel_size=4
            ),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                in_channels=24, out_channels=48, kernel_size=4
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=48, out_channels=96, kernel_size=4
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=96, out_channels=96, kernel_size=4
            ),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(
                in_channels=96, out_channels=120, kernel_size=4
            ),
            nn.BatchNorm2d(120),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=120,
                out_channels=96,
                kernel_size=4,
                output_padding=0
            ),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(
                in_channels=96,
                out_channels=96,
                kernel_size=4,
                output_padding=0,
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.ConvTranspose2d(
                in_channels=96,
                out_channels=48,
                kernel_size=4,
                output_padding=0,
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels=48,
                out_channels=24,
                kernel_size=4,
                output_padding=0
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels=24,
                out_channels=1,
                kernel_size=4,
                output_padding=0
            )
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
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z = self.encoder(val_batch)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss, logger=True, on_epoch=True, sync_dist=True)
        return x_hat

    def test_step(self, test_batch, batch_idx):
        x = test_batch
        z = self.encoder(test_batch)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss, logger=True, on_epoch=True, sync_dist=True)
        return x_hat
