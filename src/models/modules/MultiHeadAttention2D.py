import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention2D(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, mask=False):
        super(MultiheadAttention2D, self).__init__()
        self.mask = None
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.k_linear = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.v_linear = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.out_linear = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.scaling = float(self.head_dim) ** -0.5
        
    def create_mask(self, height, width):
        mask = torch.triu(torch.ones(height, width), diagonal=1).bool()
        return mask
    
    def forward(self, x):
        # x: (batch_size, in_channels, height, width) 
        _, _, h, w = x.size()

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split the channels into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # if h != w :
        #     # Reshape Q and K for matrix multiplication
        #     q = q.view(q.shape[0], q.shape[1], q.shape[2], -1)  # (batch_size, num_heads, head_dim, height*width)
        #     k = k.view(k.shape[0], k.shape[1], k.shape[2], -1).permute(0, 1, 3, 2)  # (batch_size, num_heads, head_dim, height*width)

        # Compute the dot product between queries and keys
        attn_scores = torch.matmul(q, k) * self.scaling

        if self.mask is not None:
            attn_scores = attn_scores.masked_fill(self.create_mask(h,w) == 0, float('-inf'))

        # Apply softmax to obtain attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute the weighted sum of the values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate the heads and apply the output linear layer
        attn_output = self.concat_heads(attn_output)
        attn_output = self.out_linear(attn_output)

        return attn_output

    def split_heads(self, x):
        # x: (batch_size, embed_dim, height, width)
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, self.num_heads, self.head_dim, height, width)
        return x

    def concat_heads(self, x):
        # x: (batch_size, num_heads, head_dim, height, width)
        batch_size, _, _, height, width = x.size()
        x = x.view(batch_size, -1, height, width)
        return x
