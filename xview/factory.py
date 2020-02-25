from multiprocessing.pool import Pool
from typing import List, Dict

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask, d4_image2mask
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy, rgb_image_from_tensor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .dataset import INPUT_IMAGE_KEY, INPUT_MASK_KEY, INPUT_IMAGE_ID_KEY, OUTPUT_MASK_KEY


class InMemoryDataset(Dataset):
    def __init__(self, data: List[Dict], transform: A.Compose):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(**self.data[item])


def _tensor_from_rgb_image(image: np.ndarray, **kwargs):
    return tensor_from_rgb_image(image)


class PickModelOutput(nn.Module):
    def __init__(self, model, key):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, input):
        output = self.model(input)
        return output[self.target_key]


@torch.no_grad()
def predict(model: nn.Module, image: np.ndarray, image_size, normalize=A.Normalize(), batch_size=1) -> np.ndarray:

    tile_step = (image_size[0] // 2, image_size[1] // 2)

    tile_slicer = ImageSlicer(image.shape, image_size, tile_step)
    tile_merger = CudaTileMerger(tile_slicer.target_shape, 1, tile_slicer.weight)
    patches = tile_slicer.split(image)

    transform = A.Compose([normalize, A.Lambda(image=_tensor_from_rgb_image)])

    data = list(
        {"image": patch, "coords": np.array(coords, dtype=np.int)}
        for (patch, coords) in zip(patches, tile_slicer.crops)
    )
    for batch in DataLoader(InMemoryDataset(data, transform), pin_memory=True, batch_size=batch_size):
        image = batch["image"].cuda(non_blocking=True)
        coords = batch["coords"]
        mask_batch = model(image)
        tile_merger.integrate_batch(mask_batch, coords)

    mask = tile_merger.merge()

    mask = np.moveaxis(to_numpy(mask), 0, -1)
    mask = tile_slicer.crop_to_orignal_size(mask)

    return mask


def __compute_ious(args):
    thresholds = np.arange(0, 256)
    gt, pred = args
    gt = cv2.imread(gt) > 0  # Make binary {0,1}
    pred = cv2.imread(pred)

    pred_i = np.zeros_like(gt)

    intersection = np.zeros(len(thresholds))
    union = np.zeros(len(thresholds))

    gt_sum = gt.sum()
    for index, threshold in enumerate(thresholds):
        np.greater(pred, threshold, out=pred_i)
        union[index] += gt_sum + pred_i.sum()

        np.logical_and(gt, pred_i, out=pred_i)
        intersection[index] += pred_i.sum()

    return intersection, union


def optimize_threshold(gt_images, pred_images):
    thresholds = np.arange(0, 256)

    intersection = np.zeros(len(thresholds))
    union = np.zeros(len(thresholds))

    with Pool(32) as wp:
        for i, u in tqdm(wp.imap_unordered(__compute_ious, zip(gt_images, pred_images)), total=len(gt_images)):
            intersection += i
            union += u

    return thresholds, intersection / (union - intersection)


def visualize_inria_predictions(
    input: dict,
    output: dict,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    input_image_key=INPUT_IMAGE_KEY,
    input_mask_key=INPUT_MASK_KEY,
    input_image_id_key=INPUT_IMAGE_ID_KEY,
    output_mask_key=OUTPUT_MASK_KEY,
):
    images = []
    for image, target, image_id, logits in zip(
        input[input_image_key], input[input_mask_key], input[input_image_id_key], output[output_mask_key]
    ):
        image = rgb_image_from_tensor(image, mean, std)
        target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits).squeeze(0)

        overlay = np.zeros_like(image)
        true_mask = target > 0
        pred_mask = logits > 0

        overlay[true_mask & pred_mask] = np.array(
            [0, 250, 0], dtype=overlay.dtype
        )  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array(
            [250, 250, 0], dtype=overlay.dtype
        )  # False alarm painted with yellow

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images
