import copy

import torch.nn as nn

losses = {}

def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls
    return decorator


def make(loss_spec, args=None):
    loss_args = copy.deepcopy(loss_spec['args'])
    if type(loss_args) is list:
        if args is not None:
            for i in range(len(loss_args)):
                loss_args[i]['args'].update(args)
        else:
            loss_args = loss_spec['args']
        loss = losses[loss_spec['name']](*loss_args)
    else:
        if args is not None:
            loss_args.update(args)
        else:
            loss_args = loss_spec['args']
        loss = losses[loss_spec['name']](**loss_args)

    return loss


@register('compose')
class Compose(nn.Module):
    def __init__(self, *args):
        super().__init__()

        self.fn = {}
        for arg in args:
            self.fn[arg['name']] = make(arg)

        self.fn = nn.ModuleDict(self.fn)

    def forward(self, pred, gt):
        total_loss = 0
        loss_dict = {}
        for name, fn in self.fn.items():
            loss, _ = fn(pred, gt)
            total_loss += loss
            loss_dict[name] = loss
        return total_loss, loss_dict