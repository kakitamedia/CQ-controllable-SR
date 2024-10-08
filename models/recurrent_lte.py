import torch
from torch import nn

from utils import fourier_decoding
from models import register, make

@register('recurrent-lte')
class RecurrentLTE(nn.Module):
    def __init__(self, in_dim, out_dim, rnn_spec, num_pred, block_size=1):
        super().__init__()

        self.num_pred = num_pred
        self.out_dim = out_dim
        self.block_size = block_size

        dim = 4 * out_dim * block_size
        self.dim = dim

        self.layer = make(rnn_spec, args={'in_dim': dim, 'out_dim': dim, 'embed_dim': in_dim})

        self.phase_input = nn.Conv1d(dim + 2, in_dim, kernel_size=1, stride=1, padding=0)
        self.phase = nn.Conv1d(in_dim, out_dim * block_size, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, cell, coord, *args):
        feat = feat.unsqueeze(-1)

        if self.num_pred == 0:
            return {
                'decoded': torch.zeros([feat.shape[0], 0], device=feat.device, dtype=feat.dtype),
                'coef': torch.zeros([feat.shape[0], 0], device=feat.device, dtype=feat.dtype),
                'freq': torch.zeros([feat.shape[0], 0], device=feat.device, dtype=feat.dtype),
                'phase': torch.zeros([feat.shape[0], 0], device=feat.device, dtype=feat.dtype),
            }

        fourier = self.layer.recurrent(token=feat, num_pred=self.num_pred)
        phase = self.phase(feat + self.phase_input(torch.cat(\
            [fourier, cell.unsqueeze(-1).expand(-1, -1, fourier.shape[-1])], dim=-2)))
        coef, freq, = self.unreshape(fourier)
        phase = phase.view(phase.shape[:-2] + (-1,))

        decoded = fourier_decoding(coef, freq, phase, coord)

        return {
            'decoded': decoded,
            'coef': coef,
            'freq': freq,
            'phase': phase,
        }

    def reshape(self, coef, freq):
        coef = torch.stack(torch.split(coef, 6*self.block_size, dim=-1), dim=-1)
        freq = torch.stack(torch.split(freq, 6*self.block_size, dim=-1), dim=-1)

        return torch.cat((coef, freq), dim=-2)

    def unreshape(self, fourier):
        coef, freq = torch.split(fourier, fourier.shape[-2] // 2, dim=-2)
        coef = torch.stack(torch.split(coef, 1, dim=-2), dim=-1).view(coef.shape[:-2] + (-1,))
        freq = torch.stack(torch.split(freq, 1, dim=-2), dim=-1).view(freq.shape[:-2] + (-1,))
        return coef, freq
        # return coef.view(coef.shape[:-2] + (-1,)), freq.view(coef.shape[:-2] + (-1,))

@register('channeled-recurrent-lte')
class ChanneledRecurrentLTE(nn.Module):
    def __init__(self, in_dim, out_dim, rnn_spec, num_pred=128, n_condition=False):
        super().__init__()

        self.out_dim = out_dim
        self.num_pred = num_pred

        self.layer = make(rnn_spec, args={'in_dim': 4+out_dim, 'out_dim': 4+out_dim, 'embed_dim': in_dim})

        self.phase_input = nn.Conv1d(4 + out_dim + 2, in_dim, kernel_size=1, stride=1, padding=0)
        self.phase = nn.Conv1d(in_dim, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, cell, coord, *args):
        feat = feat.unsqueeze(-1)

        fourier = self.layer.recurrent(token=feat, num_pred=self.num_pred)
        phase = self.phase(feat + self.phase_input(torch.cat(\
            [fourier, cell.unsqueeze(-1).expand(-1, -1, fourier.shape[-1])], dim=-2)))

        coef, freq, class_id = self.unreshape(fourier)
        class_id = torch.argmax(class_id, dim=-2)
        phase = phase.view(phase.shape[:-2] + (-1,))

        pred = fourier_decoding(coef, freq, phase, coord)

        return {
            'pred': pred,
            'coef': coef,
            'freq': freq,
            'phase': phase,
            'class_id': class_id,
        }

    def reshape(self, coef, freq, class_id):
        coef = torch.stack(torch.split(coef, 2, dim=-1), dim=-1)
        freq = torch.stack(torch.split(freq, 2, dim=-1), dim=-1)

        return torch.cat((coef, freq, class_id), dim=-2)

    def unreshape(self, fourier):
        coef_freq = fourier[:, :-self.out_dim]
        class_id = fourier[:, -self.out_dim:]

        coef, freq = torch.split(coef_freq, coef_freq.shape[-2] // 2, dim=-2)

        return coef.view(coef.shape[:-2] + (-1,)), freq.view(coef.shape[:-2] + (-1,)), class_id