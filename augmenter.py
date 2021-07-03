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


albumentations_function = strong_aug(p=0.9)


def augment(image):
    return albumentations_function(image=image)["image"]
