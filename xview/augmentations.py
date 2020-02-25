from typing import Tuple

import albumentations as A
import cv2

__all__ = [
    "safe_color_augmentations",
    "safe_spatial_augmentations",
    "light_color_augmentations",
    "light_spatial_augmentations",
    "light_post_image_transform",
    "medium_color_augmentations",
    "medium_spatial_augmentations",
    "medium_post_transform_augs",
    "hard_spatial_augmentations",
    "hard_color_augmentations",
    "old_light_augmentations",
    "old_post_transform_augs"
]


def safe_color_augmentations():
    return A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True)])


def safe_spatial_augmentations(image_size: Tuple[int, int]):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.MaskDropout(10),
            A.Compose([A.Transpose(), A.RandomRotate90()]),
        ]
    )


def light_color_augmentations():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True),
            A.RandomGamma(gamma_limit=(90, 110)),
        ]
    )


def light_spatial_augmentations(image_size: Tuple[int, int]):
    return A.Compose(
        [
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.RandomBrightnessContrast(),

            A.MaskDropout(max_objects=10),
        ]
    )


def old_light_augmentations(image_size: Tuple[int, int]):
    return A.Compose(
        [
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.RandomBrightnessContrast(),
            A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]
    )


def light_post_image_transform():
    return A.OneOf(
        [
            A.NoOp(),
            A.Compose(
                [
                    A.PadIfNeeded(1024 + 10, 1024 + 10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomSizedCrop((1024 - 5, 1024 + 5), 1024, 1024),
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                rotate_limit=3,
                scale_limit=0.02,
                border_mode=cv2.BORDER_CONSTANT,
                mask_value=0,
                value=0,
                p=0.2,
            ),
        ]
    )


def old_post_transform_augs():
    return A.OneOf(
        [
            A.NoOp(),
            A.Compose(
                [
                    A.PadIfNeeded(1024 + 20, 1024 + 20, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.RandomSizedCrop((1024 - 10, 1024 + 10), 1024, 1024),
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, rotate_limit=3, scale_limit=0.05, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2
            ),
        ]
    )


def medium_post_transform_augs():
    return A.OneOf(
        [
            A.NoOp(),
            A.Compose(
                [
                    A.PadIfNeeded(1024 + 40, 1024 + 40, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.RandomSizedCrop((1024 - 20, 1024 + 20), 1024, 1024),
                ]
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, rotate_limit=5, scale_limit=0.075, border_mode=cv2.BORDER_CONSTANT, value=0
            ),
        ]
    )


def medium_color_augmentations():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True),
            A.RandomGamma(gamma_limit=(90, 110)),
            A.OneOf(
                [
                    A.NoOp(p=0.8),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                ],
                p=0.2,
            ),
        ]
    )


def medium_spatial_augmentations(image_size: Tuple[int, int], no_mask_dropout=False):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.NoOp(p=0.8),
                    A.RandomGridShuffle(grid=(4, 4), p=0.2),
                    A.RandomGridShuffle(grid=(3, 3), p=0.2),
                    A.RandomGridShuffle(grid=(2, 2), p=0.2),
                ], p=1
            ),

            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            # D4 Augmentations
            A.Compose([A.Transpose(), A.RandomRotate90()]),
            # Spatial-preserving augmentations:
            A.RandomBrightnessContrast(),

            A.NoOp() if no_mask_dropout else A.MaskDropout(max_objects=10),
        ]
    )



def hard_color_augmentations():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True),
            A.RandomGamma(gamma_limit=(90, 110)),
            A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
            A.OneOf([A.RGBShift(), A.HueSaturationValue(), A.NoOp()]),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.3),
        ]
    )


def hard_spatial_augmentations(image_size: Tuple[int, int], rot_angle=45):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.NoOp(),
                    A.RandomGridShuffle(grid=(4, 4)),
                    A.RandomGridShuffle(grid=(3, 3)),
                    A.RandomGridShuffle(grid=(2, 2)),
                ]
            ),
            A.MaskDropout(max_objects=10),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        scale_limit=0.1, rotate_limit=rot_angle, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                    ),
                    A.NoOp(),
                ]
            ),
            A.OneOf(
                [
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.NoOp(),
                ]
            ),
            # D4
            A.Compose([A.Transpose(), A.RandomRotate90()]),
        ]
    )
