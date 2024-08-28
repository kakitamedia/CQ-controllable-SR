import copy
import albumentations as A

augmentations = {
    'h_flip': A.HorizontalFlip,
    'v_flip': A.VerticalFlip,
    'rot': A.RandomRotate90,
}

def register(name):
    def decorator(cls):
        augmentations[name] = cls
        return cls
    return decorator

def make(aug_spec, args=None):
    if args is not None:
        aug_args = copy.deepcopy(aug_spec['args'])
        aug_args.update(args)
    else:
        aug_args = aug_spec['args']

    if type(aug_args) is list:
        aug = augmentations[aug_spec['name']](*aug_args)
    else:
        aug = augmentations[aug_spec['name']](**aug_args)
    return aug

@register('compose')
class Compose:
    def __init__(self, *args):
        self.augs = [make(arg) for arg in args]
        self.compose = A.Compose(self.augs)

    def __call__(self, image=None):
        return self.compose(image=image)