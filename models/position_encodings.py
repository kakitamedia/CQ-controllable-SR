import torch
import torch.nn as nn

import math
from models import register

@register('sinusoidal-position-encoding')
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, q, k, pos=None):
        if pos is not None:
            pos_q = self.pe[:, pos].expand(*q.size())
            pos_k = self.pe[:, pos].expand(*k.size())
        else:
            pos_q = self.pe[:, :q.size(1)].expand(*q.size())
            pos_k = self.pe[:, :k.size(1)].expand(*k.size())

        return q + pos_q, k + pos_k

@register('learned-position-encoding')
class LearnedPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, dim))

    def forward(self, q, k, pos=None):
        if pos is not None:
            pos_q = self.pe[pos].expand(*q.size())
            pos_k = self.pe[pos].expand(*k.size())
        else:
            pos_q = self.pe[:q.size(1)].expand(*q.size())
            pos_k = self.pe[:k.size(1)].expand(*k.size())

        return q + pos_q, k + pos_k

import spe
@register('stochastic-position-encoding')
class StochasticPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()
        self.max_len = max_len
        self.spe_encoder = spe.SineSPE(num_heads=1, in_features=dim, num_realizations=dim, num_sines=5)
        self.spe_filter = spe.SPEFilter(gated=True, code_shape=self.spe_encoder.code_shape)

    def forward(self, q, k, pos=None):
        batch, seq_length, dim = q.shape

        if pos is not None:
            if pos == 0:
                self.pos_code = self.spe_encoder((batch, self.max_len))
            pos_code = (self.pos_code[0][:, pos:pos+1], self.pos_code[1][:, pos:pos+1])

        # unsqueeze to add the head dimension
        q, k = q.unsqueeze(2), k.unsqueeze(2)

        q, k = self.spe_filter(q, k, pos_code)

        return q[:, :, 0], k[:, :, 0]

if __name__ == '__main__':
    pe = SinsodiusPositionEncoding(512)
    x = torch.zeros(1, 100, 512)
    y = pe(x)
    print(y)
    print(y.shape)