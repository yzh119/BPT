import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dim_model = dim_model
        self.pe = nn.Embedding(max_len, dim_model)

    def forward(self, pos):
        return self.pe(pos)

class Embedding(nn.Module):
    "Word Embedding module"
    def __init__(self, vocab_size, dim_model):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.lut(x)

class KDEmbedding(nn.Module):
    """K-dimensional positional embedding used in OpenAI Sparse Transformer"""
    def __init__(self, dim_model, k, max_len=512):
        super(KDEmbedding, self).__init__()
        self.dim_model = dim_model
        self.k = k
        self.pe = nn.ModuleList([
            nn.Embedding(max_len, dim_model) for _ in range(k)
        ])

    def forward(self, *pos):
        rst = 0
        for p, pe in zip(pos, self.pe):
            rst = rst + pe(p)
        return rst
