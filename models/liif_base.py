import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models import register, make
from utils import make_coord

@register('liif-base')
class LIIFBase(nn.Module):
    def __init__(self, encoder_spec, predictor_spec, decoder_spec, feat_unfold=False, local_ensemble=False, residual=True):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.residual = residual

        self.encoder = make(encoder_spec)
        self.decoder = make(decoder_spec)
        self.predictor = make(predictor_spec, args={'in_dim': self.encoder.out_dim})

        self.num_pred = self.predictor.num_pred if hasattr(self.predictor, 'num_pred') else 1

    def forward(self, inp, coord, cell=None, **kwargs):
        self.gen_feat(inp)
        return self.query(coord, cell)

    def gen_feat(self, x):
        self.inp = x
        self.feat = self.encoder(x)
        self.feat_coord = make_coord(x.shape[-2:], flatten=False).to(x.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], 2, *x.shape[-2:])

    def query(self, coord, cell=None):
        feat = self.feat
        feat_coord = self.feat_coord

        # fold the 8 neighboring features into the center feature
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(\
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # setting up the local ensemble
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds, areas, miscs = [], [], {}
        for vx, vy in itertools.product(vx_lst, vy_lst):
            coord_ = coord.clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_feat  = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), \
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), \
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]

            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]

            bs, q = q_feat.shape[:2]
            q_feat, rel_cell, rel_coord = q_feat.reshape(bs * q, -1), rel_cell.reshape(bs * q, -1), rel_coord.reshape(bs * q, -1)

            latent = self.predictor(q_feat, rel_cell, rel_coord)
            out = self.decoder(latent, scale=self.num_pred)
            # out = self.decoder(latent)

            q_feat, rel_cell, rel_coord = q_feat.view(bs, q, -1), rel_cell.view(bs, q, -1), rel_coord.view(bs, q, -1)
            out = {k: v.view(bs, q, *v.shape[1:]) for k, v in out.items()}

            preds.append(out['pred'])
            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            areas.append(area + 1e-9)

            out['rel_coord'] = rel_coord
            for k, v in out.items():
                if k not in miscs:
                    miscs[k] = []
                miscs[k].append(v)

        ret = self.reconstruct_pixels(preds, areas, coord)
        miscs['inp'] = self.inp
        miscs['area'] = areas
        miscs['coord'] = coord

        return {'recon': ret, **miscs}

    def reconstruct_pixels(self, preds, areas, coord):
        total_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret += pred * (area / total_area).unsqueeze(-1)

        if self.residual:
            residual = F.grid_sample(self.inp.to(coord.device), coord.flip(-1).unsqueeze(1), \
                mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

            # when num_pred == 0, return the input
            if ret.shape[-1] == 0:
                ret = residual
            else:
                ret += residual

        return ret