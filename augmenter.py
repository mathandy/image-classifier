from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, GaussNoise, MotionBlur, MedianBlur,
    PiecewiseAffine, Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


def strong_aug(p=0.5):
    """For for examples, see
    https://github.com/albumentations-team/albumentations_examples
    """
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            PiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


autoaugment_policies = {
    'cifar10': CIFAR10Policy,
    'svhn': SVHNPolicy,
    'imagenet': ImageNetPolicy,
    'sub': SubPolicy,
}
autoaugment_choices = [f'auto-{k}' for k in autoaugment_policies.keys()]
augmentation_choices = ['none', 'strong', 'auto'] + autoaugment_choices


def get_augmentation_pipeline(args):
    if args.augmentation.lower() == 'strong':
        return lambda image: strong_aug(p=0.9)(image=image)["image"]
    elif args.augmentation.lower() in autoaugment_choices:
        policy = args.augmentation.lower().replace('auto-', '')
        return autoaugment_policies[policy]
    elif args.augmentation.lower() == 'none':
        return None
    else:
        raise ValueError(f'Augmentation mode {args.augmentation} not understood.')
