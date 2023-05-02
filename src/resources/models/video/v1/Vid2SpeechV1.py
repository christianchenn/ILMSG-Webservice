import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.functional import F

from src.models.modules.ConvLSTMCell import ConvLSTMCell


# Untuk Channel 25

class Vid2SpeechV1(pl.LightningModule):
    def __init__(self, run_name, learning_rate=1e-4, yaml_file = None):
        super().__init__()
        self.save_hyperparameters()
        self.run_name = run_name
        self.learning_rate = learning_rate
        self.yaml_file = yaml_file
        self.future_len = 25
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(
                in_channels=16,
                out_channels=1,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.conv_lstm_encoder_input = ConvLSTMCell(input_dim=1,
                                                    hidden_dim=16,
                                                    kernel_size=(3, 3),
                                                    bias=True)
        self.conv_lstm_decoder_output = ConvLSTMCell(input_dim=16,
                                                     hidden_dim=16,
                                                     kernel_size=(3, 3),
                                                     bias=True)
        # Encoders
        self.conv_lstm_encoders = []
        self.conv_lstm_encoders.append(self.conv_lstm_encoder_input)
        self.conv_lstm_encoders.append(
            ConvLSTMCell(input_dim=16,
                         hidden_dim=16,
                         kernel_size=(3, 3),
                         bias=True)
        )
        self.conv_lstm_encoders.append(
            ConvLSTMCell(input_dim=16,
                         hidden_dim=16,
                         kernel_size=(3, 3),
                         bias=True)
        )

        # Decoders
        self.conv_lstm_decoders = []
        self.conv_lstm_decoders.append(
            ConvLSTMCell(input_dim=16,
                         hidden_dim=16,
                         kernel_size=(3, 3),
                         bias=True)
        )
        self.conv_lstm_decoders.append(
            ConvLSTMCell(input_dim=16,
                         hidden_dim=16,
                         kernel_size=(3, 3),
                         bias=True)
        )
        self.conv_lstm_decoders.append(self.conv_lstm_decoder_output)

        # CNN Decoder
        self.conv_decoders = nn.Sequential(
            # 192x192 -> 64x64
            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=(1, 3, 3),
                      stride=(1, 3, 3)
                      ),
            nn.ReLU(),
            # 64x64 -> 21x21
            nn.Conv3d(in_channels=32,
                      out_channels=64,
                      kernel_size=(1, 3, 3),
                      stride=(1, 3, 3)
                      ),
            nn.ReLU(),
            # 21x21 -> 7x7
            nn.Conv3d(in_channels=64,
                      out_channels=64,
                      kernel_size=(1, 3, 3),
                      stride=(1, 2, 2)
                      ),
            nn.ReLU(),
            # 7x7 -> 4x4
            nn.Conv3d(in_channels=64,
                      out_channels=118,
                      kernel_size=(1, 3, 3),
                      stride=(1, 2, 2)
                      ),
        )

    def forward(self, x):
        # print("BEFORE FEATURE", x.size())
        # print(len(x))
        x = x.permute(0, 2, 1, 3, 4)
        # print("PERMUTE FEATURE", x.size())
        # x = self.feature_extractor(x).cuda()
        x = self.feature_extractor(x)
        # print("OUTPUT FEATURE", x.size())
        # find size of different input dimensions
        batch_size, n_channels, num_frames, h, w = x.size()
        hts, cts = self.init_hidden_states(x)

        # autoencoder forward
        outputs = self.conv_autoencoder(x, hts, cts)
        outputs = torch.stack(outputs, 1)
        # print("Before ")
        # print(outputs.size())
        outputs = outputs.permute(0, 2, 1, 3, 4)
        # print("After")
        # print(outputs.size())

        decoder_output = self.conv_decoders(outputs)
        # print("DECODER OUTPUT", decoder_output.shape)
        output = decoder_output.view(decoder_output.shape[0], decoder_output.shape[1], decoder_output.shape[2], 16)
        output = output.permute(0, 2, 1, 3)
        # print("MOD OUTPUT", output.shape)

        return output

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
        batch_size, n_channels, num_frames, h, w = x.size()
        # Batch_size, Num Frames, n_channles, w, h
        x = x.permute(0, 2, 1, 3, 4)
        for time_frame in range(num_frames):
            for i, encoder in enumerate(self.conv_lstm_encoders):
                # print(f"Iteration: {i}")
                input_tensor = x[:, time_frame, :, :, :] if i == 0 else hts[i - 1]
                # input_tensor = input_tensor.cuda()
                # print("INPUT TENSOR", input_tensor.size())
                h_t, c_t = encoder(input_tensor=input_tensor, cur_state=[hts[i], cts[i]])

                # print("HT", h_t.size())
                # print("CT", c_t.size())
                hts[i] = h_t
                cts[i] = c_t

        encoded_vector = hts[len(self.conv_lstm_encoders) - 1]
        num = len(self.conv_lstm_encoders)
        outputs = []
        for j in range(self.future_len):
            for i, decoder in enumerate(self.conv_lstm_decoders):
                input_tensor = encoded_vector if i == 0 else hts[num + i - 1]
                h_t, c_t = decoder(input_tensor=input_tensor, cur_state=[hts[num + i], cts[num + i]])
                hts[num + i] = h_t
                cts[num + i] = c_t
                if i == len(self.conv_lstm_decoders) - 1:
                    outputs.append(h_t)

        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, logger=True, on_epoch=True, sync_dist=True)
        return y_hat

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)
        self.log("test_loss", test_loss, logger=True, on_epoch=True, sync_dist=True)
        return y_hat
