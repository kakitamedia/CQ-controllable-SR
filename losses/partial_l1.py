import torch
from torch import nn

from losses import register, make
import utils

from models.recurrent_lte import RecurrentLTE

@register('partial-l1')
class PartialL1(nn.Module):
    def __init__(self, weight=1.0, log_key='partial_l1', **kwargs):
        super().__init__()

        self.fn = nn.L1Loss()
        self.weight = weight
        self.log_key = log_key
        # self.target_key = kwargs['target_key']

        self.model = kwargs['model']
        self.mode = kwargs['mode']
        self.length = kwargs.get('length')

        if self.mode == 'specify':
            assert self.length is not None

        self.num_pred = kwargs['model'].predictor.num_pred

    def forward(self, pred, gt):
        if self.mode == 'specify':
            ret = partial_reconstruction(pred, self.model, self.mode, self.num_pred, self.length)
        elif self.mode == 'random':
            ret = partial_reconstruction(pred, self.model, self.mode, self.num_pred)
        else:
            raise ValueError('Invalid mode. Please specify either "random" or "specify".')

        loss = self.fn(ret, gt['gt_rgb'])
        loss_dict = {self.log_key: loss}

        return loss * self.weight, loss_dict

def partial_reconstruction(pred, model, mode, num_pred, length=None):
    bs, q = pred['coef'][0].shape[:2]
    out_dim = model.predictor.out_dim
    block_size = getattr(model.predictor, 'block_size', 1)

    preds = []
    for i in range(len(pred['coef'])):
        coef, freq = pred['coef'][i].view(bs * q, -1), pred['freq'][i].view(bs * q, -1)
        phase, rel_coord = pred['phase'][i].view(bs * q, -1), pred['rel_coord'][i].view(bs * q, -1)

        fourier = model.predictor.reshape(coef, freq)

        if mode == 'random':
            length = torch.randint(num_pred, (bs * q, 1), device=coef.device) + 1
        elif mode == 'specify':
            length = length
        mask = torch.arange(num_pred, device=coef.device).expand(bs * q, num_pred) < length
        mask = mask.unsqueeze(-2).expand(bs * q, 2*6*block_size, num_pred)

        fourier = fourier * mask
        coef, freq = model.predictor.unreshape(fourier)

        decoded = utils.fourier_decoding(coef, freq, phase, rel_coord)
        preds.append(model.decoder({'decoded': decoded}, scale=num_pred)['pred'].view(bs, q, -1))

    ret = model.reconstruct_pixels(preds, pred['area'], pred['coord'])
    return ret