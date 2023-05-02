import pytorch_lightning as pl
import torch
from torch import nn
from torch.functional import F
from torchvision.models.video import MC3_18_Weights, mc3_18
import wandb
from src.models.modules.ConvLSTM import ConvLSTM
from src.models.modules.ConvLSTMCell import ConvLSTMCell
from src.models.modules.MultiHeadAttention2D import MultiheadAttention2D
from src.utils.model import extract_layers, extract_model, freeze_layers
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights



# Input 128x128 Transform 
# Menggunakan Attention
# Mengunakan power to db
# Target Audio (,2048)
class Vid2SpeechV41(pl.LightningModule):
    def __init__(self, run_name, learning_rate=1e-4, yaml_file=None):
        super().__init__()
        self.save_hyperparameters()
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.yaml_file = yaml_file
        self.future_len = 20 
        self.scores = {}
        self.validation_table = wandb.Table(columns=["Mel Spectrogram Image", "Original Sound", "Predicted Sound", "PESQ", "STOI", "ESTOI"])
        self.testing_table = wandb.Table(columns=["Mel Spectrogram Image", "Original Sound", "Predicted Sound", "PESQ", "STOI", "ESTOI"])
        pretrained_model = extract_layers(efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features, 0, 6)
        self.pretrained_model = freeze_layers(pretrained_model, "5.5.block.0.0.weight")
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=3,
                kernel_size=(1, 3, 3),
            ),
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )
        self.attention_1 = MultiheadAttention2D(in_channels=160, embed_dim=256, num_heads=4, mask=None)
        # self.attention_2 = MultiheadAttention2D(in_channels=256, embed_dim=256, num_heads=4, mask=None)        
        self.conv_lstm_1 = ConvLSTM(in_channel=256,
                                    hidden_dim=64,
                                    out_channel=128,
                                    future_len=self.future_len,
                                    num_decoders=2,
                                    num_encoders=2)
        
        # CNN Decoder
        self.conv_decoders = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=64,
                kernel_size=(1,3,3),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1,3,3),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32,
                      out_channels=16,
                      kernel_size=(1,1,1),
                      ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16,
                      out_channels=8,
                      kernel_size=(1,1,1),
                      ),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 2048),
            # nn.Linear(3072, 2048)
        )


    def forward(self, x):
        batch_size, num_frames, channels, h, w = x.size()
        # B, NF, C, H, W -> B, C, NF, H, W
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.feature_extractor(x)
        
        features = []
        for time_frame in range(num_frames):
            feature = self.pretrained_model(x[:, :, time_frame, :, :])
            features.append(feature)
        x = torch.stack(features, 2)
        
        # print("BEFORE ATTENTION", x.size())
        attention_features = []
        # print("SIZE BEFORE", x.size())
        for time_frame in range(num_frames):
            feature = self.attention_1(x[:, :, time_frame, :, :])
            attention_features.append(feature)
        x = torch.stack(attention_features, 2)
        
        x = self.conv_lstm_1(x)
        x = torch.stack(x, 2)
        
        # Decoder
        # B, C, NF, H, W
        decoder_output = self.conv_decoders(x)
        
        return decoder_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def calculate_loss(self, input, target):
        total_loss = torch.zeros(1, device=input.device)
        for i in range(input.size(1)):
            loss = F.mse_loss(input[:, i,:,:], target[:, i,:,:])
            total_loss+=loss
        return total_loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        # B, C, NF, HW -> B, NF, C, HW 
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        # print("Y SHAPE", y.size())
        y = y.squeeze()
        y_hat = self(x).squeeze()
        # B, C, NF, HW -> B, NF, C, HW 
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, logger=True, on_epoch=True, sync_dist=True)
        return y_hat

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        # B, C, NF, HW -> B, NF, C, HW 
        test_loss = F.mse_loss(y_hat, y)
        self.log("test_loss", test_loss, logger=True, on_epoch=True, sync_dist=True)
        return y_hat
