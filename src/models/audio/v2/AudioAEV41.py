import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F
class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor
# Sepert AEV40 tetapi dengan sigmoid, sudah dicoba dengan tanh, lebih baik sigmoid harusnya
class AudioAEV41(pl.LightningModule):
    def __init__(self, learning_rate, run_name, yaml_file):
        super().__init__()
        self.save_hyperparameters()
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.scores = {}
        self.yaml_file = yaml_file
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=8, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=4, kernel_size=(3,3)
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3192, 2048),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2048, 3192),
            nn.Unflatten(-1, (4, 114, 7)),
            nn.ConvTranspose2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(3,3),
                output_padding=0
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3,3),
                output_padding=0
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3,3),
                output_padding=0
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                output_padding=0
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                output_padding=0
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                output_padding=0
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=(3,3),
                output_padding=0
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        # x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
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
