import torch
from torch import nn

from src.models.modules.ConvLSTM import ConvLSTM

class BidirectionalConvLSTM(nn.Module):
    def __init__(self, in_channel, out_channel, future_len, hidden_dim=64, num_encoders=2, num_decoders=2):
        super().__init__()

        self.forward_conv_lstm = ConvLSTM(in_channel, out_channel // 2, future_len, hidden_dim, num_encoders, num_decoders)
        self.backward_conv_lstm = ConvLSTM(in_channel, out_channel // 2, future_len, hidden_dim, num_encoders, num_decoders)

    def forward(self, x):
        forward_outputs = self.forward_conv_lstm(x)
        backward_outputs = self.backward_conv_lstm(torch.flip(x, dims=[2]))

        # Flip the backward outputs back to the original order
        backward_outputs = [torch.flip(output, dims=[2]) for output in backward_outputs]

        # Concatenate forward and backward outputs along the hidden_dim axis
        outputs = [torch.cat((f_out, b_out), dim=1) for f_out, b_out in zip(forward_outputs, backward_outputs)]

        return outputs
