import cv2
import torch
from albumentations.augmentations.functional import longest_max_size
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, to_numpy

from xview.dataset import (
    INPUT_IMAGE_PRE_KEY,
    INPUT_IMAGE_POST_KEY,
    INPUT_MASK_PRE_KEY,
    OUTPUT_MASK_PRE_KEY,
    INPUT_MASK_POST_KEY,
    OUTPUT_MASK_POST_KEY,
    INPUT_IMAGE_KEY,
    INPUT_MASK_KEY,
    INPUT_IMAGE_ID_KEY)

import numpy as np


def overlay_image_and_mask(image, mask, class_colors, alpha=0.5):
    overlay = image.copy()
    for class_index, class_color in enumerate(class_colors):
        if class_index == 0:
            continue  # Skip background
        overlay[mask == class_index, :] = class_color

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


@torch.no_grad()
def draw_predictions(
    input: dict,
    output: dict,
    image_id_key="image_id",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    class_colors=[
        (0, 0, 0),  # 0=background
        (0, 255, 0),  # no damage (or just 'building' for localization) (green)
        (255, 255, 0),  # minor damage (yellow)
        (255, 128, 0),  # major damage (red)
        (255, 0, 0),  # destroyed (red)
        (127, 127, 127)
    ],
    max_images=32
):
    images = []

    num_images = len(input[image_id_key])
    for i in range(num_images):
        image_id = input[INPUT_IMAGE_ID_KEY][i]
        image_pre = rgb_image_from_tensor(input[INPUT_IMAGE_KEY][i, 0:3, ...], mean, std)
        image_pre = cv2.cvtColor(image_pre, cv2.COLOR_RGB2BGR)

        image_post = rgb_image_from_tensor(input[INPUT_IMAGE_KEY][i, 3:6, ...], mean, std)
        image_post = cv2.cvtColor(image_post, cv2.COLOR_RGB2BGR)

        image_pre_gt = image_pre.copy()
        image_post_gt = image_post.copy()

        damage_target = to_numpy(input[INPUT_MASK_KEY][i])

        image_pre_gt = overlay_image_and_mask(image_pre_gt, damage_target, class_colors)
        image_post_gt = overlay_image_and_mask(image_post_gt, damage_target, class_colors)

        damage_predictions = to_numpy(output[INPUT_MASK_KEY][i]).argmax(axis=0)

        image_pre = overlay_image_and_mask(image_pre, damage_predictions, class_colors)
        image_post = overlay_image_and_mask(image_post, damage_predictions, class_colors)

        overlay_gt = np.column_stack([image_pre_gt, image_post_gt])
        overlay = np.column_stack([image_pre, image_post])
        overlay = np.row_stack([overlay_gt, overlay])

        overlay = longest_max_size(overlay, 1024, cv2.INTER_LINEAR)

        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))
        images.append(overlay)
        if len(images) >= max_images:
            break

    return images


@torch.no_grad()
def draw_predictions_dual(
    input: dict,
    output: dict,
    image_id_key="image_id",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    class_colors=[
        (0, 0, 0),  # 0=background
        (0, 255, 0),  # no damage (or just 'building' for localization) (green)
        (255, 255, 0),  # minor damage (yellow)
        (255, 128, 0),  # major damage (red)
        (255, 0, 0),  # destroyed (red)
    ],
):
    images = []

    num_images = len(input[image_id_key])
    for i, image_id in enumerate(range(num_images)):
        image_pre = rgb_image_from_tensor(input[INPUT_IMAGE_PRE_KEY][i], mean, std)
        image_pre = cv2.cvtColor(image_pre, cv2.COLOR_RGB2BGR)

        image_post = rgb_image_from_tensor(input[INPUT_IMAGE_POST_KEY][i], mean, std)
        image_post = cv2.cvtColor(image_post, cv2.COLOR_RGB2BGR)

        image_pre_gt = image_pre.copy()
        image_post_gt = image_post.copy()

        localization_target = to_numpy(input[INPUT_MASK_PRE_KEY][i].squeeze(0))
        damage_target = to_numpy(input[INPUT_MASK_POST_KEY][i])

        image_pre_gt = overlay_image_and_mask(image_pre_gt, localization_target, class_colors)
        image_post_gt = overlay_image_and_mask(image_post_gt, damage_target, class_colors)

        localization_predictions = to_numpy(output[OUTPUT_MASK_PRE_KEY][i].squeeze(0).sigmoid() > 0.5).astype(np.uint8)
        damage_predictions = to_numpy(output[OUTPUT_MASK_POST_KEY][i]).argmax(axis=0)

        image_pre = overlay_image_and_mask(image_pre, localization_predictions, class_colors)
        image_post = overlay_image_and_mask(image_post, damage_predictions, class_colors)

        overlay_gt = np.column_stack([image_pre_gt, image_post_gt])
        overlay = np.column_stack([image_pre, image_post])
        overlay = np.row_stack([overlay_gt, overlay])

        overlay = longest_max_size(overlay, 1024, cv2.INTER_LINEAR)

        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))
        images.append(overlay)

    return images
