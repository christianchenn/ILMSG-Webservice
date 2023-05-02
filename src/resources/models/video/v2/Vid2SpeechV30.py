import pytorch_lightning as pl
import torch
from torch import nn
from torch.functional import F
from torchvision.models.video import MC3_18_Weights, mc3_18
from src.models.modules.ConvLSTMCell import ConvLSTMCell
from src.models.modules.MultiHeadAttention2D import MultiheadAttention2D
from src.utils.model import extract_layers, extract_model, freeze_layers
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights



# Input 128x128 Transform 3
# Menggunakan Attention
# Mengunakan power to db
# Target Audio (,2048)
class Vid2SpeechV30(pl.LightningModule):
    def __init__(self, run_name, learning_rate=1e-4, yaml_file=None):
        super().__init__()
        self.save_hyperparameters()
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.yaml_file = yaml_file
        self.future_len = 20 
        self.attention = MultiheadAttention2D(in_channels=1, embed_dim=256, num_heads=4)
        # Encoders
        self.conv_lstm_encoders = []
        self.conv_lstm_encoders.append(
            ConvLSTMCell(input_dim=256,
                         hidden_dim=64,
                         kernel_size=(3, 3),
                         bias=True)
        )
        self.conv_lstm_encoders.append(
            ConvLSTMCell(input_dim=64,
                         hidden_dim=64,
                         kernel_size=(3, 3),
                         bias=True)
        )

        # Decoders
        self.conv_lstm_decoders = []
        self.conv_lstm_decoders.append(
            ConvLSTMCell(input_dim=64,
                         hidden_dim=64,
                         kernel_size=(3, 3),
                         bias=True)
        )
        self.conv_lstm_decoders.append(
            ConvLSTMCell(input_dim=64,
                         hidden_dim=128,
                         kernel_size=(3, 3),
                         bias=True)
        )
        
        # CNN Decoder
        self.conv_decoders = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=64,
                kernel_size=(4,5,5),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=64,
                kernel_size=(4,5,5),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=(4,5,5),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32,
                      out_channels=32,
                      kernel_size=(2,5,5),
                      ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32,
                      out_channels=16,
                      kernel_size=(2,5,5),
                      ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16,
                      out_channels=8,
                      kernel_size=(2,5,5),
                      ),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8,
                      out_channels=1,
                      kernel_size=(2,5,5),
                      ),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7396, 4096),
            nn.Linear(4096, 3072),
            nn.Linear(3072, 2048)
        )


    def forward(self, x):
        batch_size, num_frames, channels, h, w = x.size()
        # B, NF, C, H, W -> B, C, NF, H, W
        x = x.permute(0, 2, 1, 3, 4)
        attention_features = []
        # print("SIZE BEFORE", x.size())
        for time_frame in range(num_frames):
            feature = self.attention(x[:, :, time_frame, :, :])
            attention_features.append(feature)
        x = torch.stack(attention_features, 1)
        
        
        # find size of different input dimensions
        hts, cts = self.init_hidden_states(x)

        # ConvLSTM Autoencoder forward
        # B, NF, C, H, W 
        outputs = self.conv_autoencoder(x, hts, cts)
        outputs = torch.stack(outputs, 1)
        # print("OUTPUT SIZE", outputs.size())
        
        # Decoder
        # B, NF, C, H, W -> B, C, NF, H, W
        outputs = outputs.permute(0, 2, 1, 3, 4)
        decoder_output = self.conv_decoders(outputs)
        
        # Reshape to Target Size (B, CH, NF, 16)
        # print("DECODER SIZE", decoder_output.size())
        # print("===========")
        # print(decoder_output.max(), decoder_output.min())
        # print("===========")
        # decoder_output = decoder_output.view(decoder_output.shape[0], decoder_output.shape[1], decoder_output.shape[2], decoder_output.shape[3]*decoder_output.shape[3])
        # decoder_output = decoder_output.permute(0, 1, 3, 2)
        return decoder_output
    
    def init_hidden_states(self, x):
        batch_size, n_channels, num_frames, h, w = x.size()
        # initialize hidden states
        hts = []
        cts = []

        for encoder in self.conv_lstm_encoders:
            ht, ct = encoder.init_hidden(batch_size=batch_size, image_size=(h, w))
            hts.append(ht)
            cts.append(ct)
        for decoder in self.conv_lstm_decoders:
            ht, ct = decoder.init_hidden(batch_size=batch_size, image_size=(h, w))
            hts.append(ht)
            cts.append(ct)
        return hts, cts

    def conv_autoencoder(self, x, hts, cts):
        batch_size, num_frames, channels, h, w = x.size()
        
        for time_frame in range(num_frames):
            for i, encoder in enumerate(self.conv_lstm_encoders):
                input_tensor = x[:, time_frame, :, :, :] if i == 0 else hts[i-1]
                h_t, c_t = encoder(input_tensor=input_tensor, cur_state=[hts[i], cts[i]])
                hts[i] = h_t
                cts[i] = c_t

        encoded_vector = hts[len(self.conv_lstm_encoders)-1]
        num = len(self.conv_lstm_encoders)
        outputs = []
        for j in range(self.future_len):
          for i, decoder in enumerate(self.conv_lstm_decoders):
            input_tensor = encoded_vector if i == 0 else hts[num+i-1]
            h_t, c_t = decoder(input_tensor=input_tensor, cur_state=[hts[num+i], cts[num+i]])
            hts[num+i] = h_t
            cts[num+i] = c_t
            if i == len(self.conv_lstm_decoders)-1:
              outputs.append(h_t)

        return outputs

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
