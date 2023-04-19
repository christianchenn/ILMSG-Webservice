import pytorch_lightning as pl
import torch
from torch import nn
from torch.functional import F
from torchvision.models.video import MC3_18_Weights, mc3_18
from src.models.modules.ConvLSTMCell import ConvLSTMCell
from src.utils.model import extract_layers, extract_model, freeze_layers
from torchvision.models import vgg16, VGG16_Weights



# Input 128x128 Transform 4
# Untuk Target 25x118x16
# Menggunakan Efficientnet
# Mencoba menggunakan loss function yang baru V2
class Vid2SpeechV16(pl.LightningModule):
    def __init__(self, run_name, learning_rate=1e-4, yaml_file=None):
        super().__init__()
        self.save_hyperparameters()
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.yaml_file = yaml_file

        # python summary.py -m vgg16 --start 0 --end 30 --freeze 29.weight --type 2d --out_channels 3
        pretrained_model = extract_layers(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features, 0, 30)
        self.pretrained_model = freeze_layers(pretrained_model, "24.weight")
        self.flatten = nn.Flatten()
        self.linear_encoders = nn.Sequential(
            nn.Linear(1216, 512),
        )
        self.lstm = nn.LSTM(512, 256, 1)
        self.linear_decoders = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 400)
        )
        self.feature_extractor_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=3,
                kernel_size=(1, 3, 3),
            ),
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )
        self.feature_extractor_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3, 3),
                stride=1
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3, 3),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3, 3),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
        # CNN Decoder
        self.conv_decoders = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=32,
                kernel_size=(1, 1,1),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 1, 1),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # 192x192 -> 64x64
            nn.Conv3d(in_channels=64,
                      out_channels=118,
                      kernel_size=(1, 1, 1),
                      stride=(1,1,1)
                      ),
            nn.BatchNorm3d(118),
            nn.ReLU(),
            nn.Sigmoid(),
        )


    def forward(self, x):
        batch_size, num_frames, channels, h, w = x.size()
        # B, NF, C, H, W -> B, C, NF, H, W
        x = x.permute(0, 2, 1, 3, 4)
        x = self.feature_extractor_1(x)
        features = []
        # print("SIZE BEFORE", x.size())
        for time_frame in range(num_frames): 
            feature = self.pretrained_model(x[:, :, time_frame, :, :])
            # print("Feature", feature.size())
            features.append(feature)
        x = torch.stack(features, 2)
        # print("BEFORE EXTRACT 2", x.size())
        x = self.feature_extractor_2(x)
        
        x = self.flatten(x)
        # print("Flatten: ", x.size())
        x = self.linear_encoders(x)
        # print("SIZE: ", x.size())
        
        out, _ = self.lstm(x)
        # print("SIZE 2: ", out.size())
        x = self.linear_decoders(out)
        # print("SIZE 3: ", x.size())
        
        # Decoder
        # B, NF, C, H, W -> B, C, NF, H, W
        outputs = x.view(x.shape[0], 1, 25, 4, 4)
        decoder_output = self.conv_decoders(outputs)
        
        # Reshape to Target Size (B, CH, NF, 16)
        # print("DECODER SIZE", decoder_output.size())
        decoder_output = decoder_output.view(decoder_output.shape[0], decoder_output.shape[1], decoder_output.shape[2], 16)
        return decoder_output
    
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def caluculate_loss(self, input, target):
        total_loss = torch.zeros(1, device=input.device)
        for i in range(input.size(1)):
            loss = F.mse_loss(input[:, i,:,:], target[:, i,:,:])
            total_loss+=loss
        return total_loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        # B, C, NF, HW -> B, NF, C, HW 
        y_hat = y_hat.permute(0, 2, 1, 3)
        loss = F.mse_loss(y_hat, y, reduction="sum")/y_hat.size()[0]
        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        # B, C, NF, HW -> B, NF, C, HW 
        y_hat = y_hat.permute(0, 2, 1, 3)
        val_loss = F.mse_loss(y_hat, y, reduction="sum")
        self.log("val_loss", val_loss, logger=True, on_epoch=True, sync_dist=True)
        return y_hat

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        # B, C, NF, HW -> B, NF, C, HW 
        y_hat = y_hat.permute(0, 2, 1, 3)
        test_loss = F.mse_loss(y_hat, y, reduction="sum")
        self.log("test_loss", test_loss, logger=True, on_epoch=True, sync_dist=True)
        return y_hat
