from torch import nn
import torch

class GaussianNoiseLayer(nn.Module):
    def __init__(self, noise_level):
        super(GaussianNoiseLayer, self).__init__()
        self.noise_level = noise_level
    
    def forward(self, x):
        return self.gaussian(x)
    
    def gaussian(self, x):
        # Implement your denoising function here
        return torch.normal(0, self.noise_level, size=x.shape, device="cuda")