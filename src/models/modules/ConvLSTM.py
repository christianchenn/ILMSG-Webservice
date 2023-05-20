from torch import nn
import torch

from src.models.modules.ConvLSTMCell import ConvLSTMCell
class ConvLSTM(nn.Module):
    
    def __init__(self, in_channel, out_channel, future_len, hidden_dim=64, num_encoders=2, num_decoders=2) -> None:
       super().__init__()
       self.encoders = nn.ModuleList()
       self.decoders = nn.ModuleList()
       self.in_channel = in_channel
       self.out_channel = out_channel
       self.future_len = future_len
       self.hidden_dim = hidden_dim
       self.num_encoders = num_encoders
       self.num_decoders = num_decoders
       self.configure()
        
    def configure(self):
        self.encoders.append(
            ConvLSTMCell(input_dim=self.in_channel,
                         hidden_dim=self.hidden_dim,
                         kernel_size=(3, 3),
                         bias=True)
        )
        for i in range(self.num_encoders-1):
            self.encoders.append(
                ConvLSTMCell(input_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            kernel_size=(3, 3),
                            bias=True)
            )
        for i in range(self.num_decoders-1):
            self.decoders.append(
                ConvLSTMCell(input_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            kernel_size=(3, 3),
                            bias=True)
            )
        self.decoders.append(
            ConvLSTMCell(input_dim=self.hidden_dim,
                         hidden_dim=self.out_channel,
                         kernel_size=(3, 3),
                         bias=True)
        )
    
    def forward(self, x):
        hts, cts = self.init_hidden_states(x)
        x = self.conv_autoencoder(x, hts, cts)
        return x
    
    def init_hidden_states(self, x):
        batch_size, n_channels, num_frames, h, w = x.size()
        # initialize hidden states
        hts = []
        cts = []

        for encoder in self.encoders:
            ht, ct = encoder.init_hidden(batch_size=batch_size, image_size=(h, w))
            hts.append(ht)
            cts.append(ct)
        for decoder in self.decoders:
            ht, ct = decoder.init_hidden(batch_size=batch_size, image_size=(h, w))
            hts.append(ht)
            cts.append(ct)
        return hts, cts
    
    def conv_autoencoder(self, x, hts, cts):
        batch_size, n_channels, num_frames, h, w = x.size()
        
        for time_frame in range(num_frames):
            for i, encoder in enumerate(self.encoders):
                input_tensor = x[:, :, time_frame, :, :] if i == 0 else hts[i-1]
                h_t, c_t = encoder(input_tensor=input_tensor, cur_state=[hts[i], cts[i]])
                hts[i] = h_t
                cts[i] = c_t

        num = len(self.encoders)
        encoded_vector = hts[num-1]
        outputs = []
        for j in range(self.future_len):
          for i, decoder in enumerate(self.decoders):
            input_tensor = encoded_vector if i == 0 else hts[num+i-1]
            h_t, c_t = decoder(input_tensor=input_tensor, cur_state=[hts[num+i], cts[num+i]])
            hts[num+i] = h_t
            cts[num+i] = c_t
            if i == len(self.decoders)-1:
              outputs.append(h_t)

        return outputs