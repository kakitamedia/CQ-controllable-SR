import torch
from torch import nn

from losses import register, make

@register('fourier-gan')
class FourierGANLoss(nn.Module):
    def __init__(self, weight=1.0, log_key='fourier_gan', **kwargs):
        super().__init__()

        self.loss_fn = make(kwargs['loss_fn'])
        self.weight = weight
        self.log_key = log_key
        self.model = kwargs['model']

        dim = self.model.predictor.out_dim * self.model.predictor.num_pred * 4

        self.discriminator = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, pred, gt):
        bs, q = pred['coef'][0].shape[:2]

        gt_inp = {
            'inp': gt['gt_img'],
            'coord': gt['gt_coord'],
            'cell':gt['gt_cell'],
        }
        with torch.no_grad():
            gt_pred = self.model(**gt_inp)

        loss = 0
        for i in range(len(pred['coef'])):
            pred_fourier = self.model.predictor.reshape(pred['coef'][i], pred['freq'][i]).view(bs, q, -1)
            gt_fourier = self.model.predictor.reshape(gt_pred['coef'][i], gt_pred['freq'][i]).view(bs, q, -1)
            loss += self.loss_fn(self.discriminator(pred_fourier), self.discriminator(gt_fourier))

        loss_dict = {self.log_key: loss}

        return loss * self.weight, loss_dict

@register('hinge')
class HingeLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.relu = nn.ReLU()

    def forward(self, pred, gt):
        return (self.relu(1 + pred) + self.relu(1 - gt)).mean()

@register('wasserstain')
class WasserstainLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, gt):
        return (pred - gt).mean()