import numpy as np

import torch
from torch import nn

from utils import fourier_decoding
from models import register

@register('lte')
class LTE(nn.Module):
    def __init__(self, in_dim, out_dim, num_pred):
        super().__init__()

        self.out_dim = out_dim

        self.coef = nn.Linear(in_dim, out_dim*2*num_pred)
        self.freq = nn.Linear(in_dim, out_dim*2*num_pred)
        self.phase = nn.Linear(2, out_dim*num_pred, bias=False)

    def forward(self, feat, cell, coord, *args):
        coef = self.coef(feat)
        freq = self.freq(feat)
        phase = self.phase(cell)

        decoded = fourier_decoding(coef, freq, phase, coord)

        return {
            'decoded': decoded,
            'coef': coef,
            'freq': freq,
            'phase': phase,
        }

    def reshape(self, coef, freq):
        coef = torch.stack(torch.split(coef, 6, dim=-1), dim=-1)
        freq = torch.stack(torch.split(freq, 6, dim=-1), dim=-1)

        return torch.cat((coef, freq), dim=-2)

    def unreshape(self, fourier):
        coef, freq = torch.split(fourier, fourier.shape[-2] // 2, dim=-2)
        coef = torch.stack(torch.split(coef, 1, dim=-2), dim=-1).view(coef.shape[:-2] + (-1,))
        freq = torch.stack(torch.split(freq, 1, dim=-2), dim=-1).view(freq.shape[:-2] + (-1,))
        return coef, freq
