import os
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from scipy.ndimage import binary_dilation, binary_fill_holes
from sklearn.utils import compute_sample_weight, compute_class_weight
from torch.utils.data import Dataset, WeightedRandomSampler, ConcatDataset

from .alignment import align_post_image
from .augmentations import *

from .utils.inference_image_output import colorize_mask

INPUT_IMAGE_KEY = "image"

INPUT_IMAGE_PRE_KEY = "image_pre"
INPUT_IMAGE_POST_KEY = "image_post"

INPUT_IMAGE_ID_KEY = "image_id"
INPUT_MASK_KEY = "mask"
INPUT_MASK_PRE_KEY = "mask_pre"
INPUT_MASK_POST_KEY = "mask_post"

OUTPUT_EMBEDDING_KEY = "embedding"
OUTPUT_MASK_KEY = "mask"
OUTPUT_MASK_ARC_KEY = "mask_arc"
OUTPUT_MASK_PRE_KEY = "mask_pre"
OUTPUT_MASK_POST_KEY = "mask_post"

INPUT_INDEX_KEY = "index"
DISASTER_TYPE_KEY = "disaster_type"
DAMAGE_TYPE_KEY = "damage_type"

# Smaller masks for deep supervision
OUTPUT_MASK_4_KEY = "mask_4"
OUTPUT_MASK_8_KEY = "mask_8"
OUTPUT_MASK_16_KEY = "mask_16"
OUTPUT_MASK_32_KEY = "mask_32"

OUTPUT_CLASS_KEY = "classes"

UNLABELED_SAMPLE = 5

DAMAGE_TYPES = ["no_damage", "minor_damage", "major_damage", "destroyed"]
DISASTER_TYPES = ["volcano", "fire", "tornado", "tsunami", "flooding", "earthquake", "hurricane"]
UNKNOWN_DISASTER_TYPE_CLASS = -100


def get_disaster_class_from_fname(fname: str) -> int:
    image_id = fs.id_from_fname(fname)

    for i, disaster_name in enumerate(DISASTER_TYPES):
        if disaster_name in image_id:
            return i

    return UNKNOWN_DISASTER_TYPE_CLASS


def read_image(fname):
    image = cv2.imread(fname)
    if image is None:
        raise FileNotFoundError(fname)
    return image


def read_mask(fname):
    mask = np.array(Image.open(fname))  # Read using PIL since it supports palletted image
    if len(mask.shape) == 3:
        mask = np.squeeze(mask, axis=-1)
    return mask


def compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    dilated = binary_dilation(mask, structure=np.ones((5, 5), dtype=np.bool))
    dilated = binary_fill_holes(dilated)

    diff = dilated & ~mask
    diff = cv2.dilate(diff, kernel=(5, 5))
    diff = diff & ~mask
    return diff.astype(np.uint8)


class ImageLabelDataset(Dataset):
    def __init__(
        self,
        pre_image_filenames: List[str],
        post_image_filenames: List[str],
        targets: Optional[np.ndarray],
        spatial_transform: A.Compose,
        color_transform: A.Compose = None,
        post_image_transform=None,
        image_loader=read_image,
    ):
        assert len(pre_image_filenames) == len(post_image_filenames)

        self.pre_image_filenames = pre_image_filenames
        self.post_image_filenames = post_image_filenames

        self.targets = targets

        self.get_image = image_loader

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.post_image_transform = post_image_transform

    def __len__(self):
        return len(self.pre_image_filenames)

    def __getitem__(self, index):
        pre_image = self.get_image(self.pre_image_filenames[index])
        post_image = self.get_image(self.post_image_filenames[index])

        if self.color_transform is not None:
            pre_image = self.color_transform(image=pre_image)["image"]
            post_image = self.color_transform(image=post_image)["image"]

        if self.post_image_transform is not None:
            post_image = self.post_image_transform(image=post_image)["image"]

        image = np.dstack([pre_image, post_image])
        data = {"image": image}
        data = self.spatial_transform(**data)

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.pre_image_filenames[index]),
            INPUT_IMAGE_KEY: tensor_from_rgb_image(data["image"]),
        }

        if self.targets is not None:
            target = int(self.targets[index])
            sample[DAMAGE_TYPE_KEY] = target

        return sample


class ImageMaskDataset(Dataset, PseudolabelDatasetMixin):
    def __init__(
        self,
        pre_image_filenames: List[str],
        post_image_filenames: List[str],
        post_mask_filenames: Optional[List[str]],
        spatial_transform: A.Compose,
        color_transform: A.Compose = None,
        post_image_transform=None,
        image_loader=read_image,
        mask_loader=read_mask,
        use_edges=False,
        align_post=False,
    ):
        assert len(pre_image_filenames) == len(post_image_filenames)

        self.use_edges = use_edges

        self.pre_image_filenames = pre_image_filenames
        self.post_image_filenames = post_image_filenames
        self.post_mask_filenames = post_mask_filenames

        self.get_image = image_loader
        self.get_mask = mask_loader

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.post_image_transform = post_image_transform
        self.align_post = align_post

    def __len__(self):
        return len(self.pre_image_filenames)

    def __getitem__(self, index):
        pre_image = self.get_image(self.pre_image_filenames[index])
        post_image = self.get_image(self.post_image_filenames[index])

        if self.align_post:
            post_image = align_post_image(pre_image, post_image)

        if self.color_transform is not None:
            pre_image = self.color_transform(image=pre_image)["image"]
            post_image = self.color_transform(image=post_image)["image"]

        if self.post_image_transform is not None:
            post_image = self.post_image_transform(image=post_image)["image"]

        image = np.dstack([pre_image, post_image])
        data = {"image": image}

        if self.post_mask_filenames is not None:
            post_mask = self.get_mask(self.post_mask_filenames[index])
            # assert np.all((post_mask >= 0) & (post_mask < 5)), f"Mask for sample {index} {self.post_mask_filenames[index]} contains values {np.unique(post_mask)}"

            data["mask"] = post_mask
        else:
            data["mask"] = np.ones(image.shape[:2], dtype=int) * UNLABELED_SAMPLE

        data = self.spatial_transform(**data)

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.pre_image_filenames[index]),
            INPUT_IMAGE_KEY: tensor_from_rgb_image(data["image"]),
            DISASTER_TYPE_KEY: get_disaster_class_from_fname(self.pre_image_filenames[index]),
        }

        if "mask" in data:
            post_mask = data["mask"]
            sample[INPUT_MASK_KEY] = torch.from_numpy(post_mask).long()
            sample[DAMAGE_TYPE_KEY] = torch.tensor(
                [(post_mask == 1).any(), (post_mask == 2).any(), (post_mask == 3).any(), (post_mask == 4).any()]
            ).float()

        return sample

    def set_target(self, index: int, value: np.ndarray):
        mask_fname = self.post_mask_filenames[index]

        value = value.astype(np.uint8)
        value = colorize_mask(value)
        value.save(mask_fname)


def get_transforms(image_size, augmentation, train_on_crops, enable_post_image_transform):
    if train_on_crops:
        train_crop_or_resize = A.RandomSizedCrop(
            (int(image_size[0] * 0.8), int(image_size[0] * 1.2)), image_size[0], image_size[1]
        )
        valid_crop_or_resize = A.NoOp()
        print("Training on crops", train_crop_or_resize.min_max_height)
    else:
        if image_size[0] != 1024 or image_size[1] != 1024:
            train_crop_or_resize = A.Resize(image_size[0], image_size[1])
        else:
            train_crop_or_resize = A.NoOp()
        valid_crop_or_resize = train_crop_or_resize

    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
    )

    train_spatial_augs = A.NoOp()
    train_color_augs = None

    # This transform slightly moves post- image to simulate imperfect alignment of satellite
    post_image_transform = None

    if augmentation == "hard":
        train_color_augs = hard_color_augmentations()
        train_spatial_augs = hard_spatial_augmentations(image_size)
        post_image_transform = light_post_image_transform()
    elif augmentation == "medium":
        train_color_augs = medium_color_augmentations()
        train_spatial_augs = medium_spatial_augmentations(image_size)
        post_image_transform = medium_post_transform_augs()
    elif augmentation == "medium_nmd":
        train_color_augs = medium_color_augmentations()
        train_spatial_augs = medium_spatial_augmentations(image_size, no_mask_dropout=True)
        post_image_transform = medium_post_transform_augs()
    elif augmentation == "light":
        train_color_augs = light_color_augmentations()
        train_spatial_augs = light_spatial_augmentations(image_size)
        post_image_transform = light_post_image_transform()
    elif augmentation == "old":
        train_color_augs = None
        train_spatial_augs = old_light_augmentations(image_size)
        post_image_transform = old_post_transform_augs()
    elif augmentation == "safe":
        train_color_augs = safe_color_augmentations()
        train_spatial_augs = safe_spatial_augmentations(image_size)
        post_image_transform = old_post_transform_augs()

    train_transform = A.Compose([train_crop_or_resize, train_spatial_augs, normalize])
    valid_transform = A.Compose([valid_crop_or_resize, normalize])

    if enable_post_image_transform:
        print("Enabling post-image spatial transformation")
    else:
        post_image_transform = None

    return train_transform, train_color_augs, valid_transform, post_image_transform


def get_datasets(
    data_dir: str,
    image_size=(512, 512),
    augmentation="safe",
    use_edges=False,
    sanity_check=False,
    fast=False,
    fold=0,
    only_buildings=False,
    balance=False,
    train_on_crops=False,
    enable_post_image_transform=False,
    align_post=False,
    crops_multiplication_factor=3
):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_edges: If True, adds 'edge' target mask
    :param augmentation: Type of image augmentations to use
    :param train_mode:
    'random' - crops tiles from source images randomly.
    'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """

    df = pd.read_csv(os.path.join(data_dir, "train_folds.csv"))
    df = make_dual_dataframe(df)

    train_transform, train_color_augs, valid_transform, post_image_transform = get_transforms(
        image_size=image_size,
        augmentation=augmentation,
        train_on_crops=train_on_crops,
        enable_post_image_transform=enable_post_image_transform,
    )

    train_df = df[df["fold_post"] != fold]
    valid_df = df[df["fold_post"] == fold]

    if only_buildings:
        only_buildings_mask = train_df["non_damaged_buildings_pre"] > 0
        total = len(train_df)
        percentage = only_buildings_mask.sum() / float(total)
        train_df = train_df[only_buildings_mask]
        print("Using only images with buildings for training", percentage)

    if fast:
        train_df = train_df[:128]
        valid_df = valid_df[:128]
        train_sampler = None

    train_img_pre = [os.path.join(data_dir, fname) for fname in train_df["image_fname_pre"]]
    train_img_post = [os.path.join(data_dir, fname) for fname in train_df["image_fname_post"]]
    train_mask_post = [os.path.join(data_dir, fname) for fname in train_df["mask_fname_post"]]

    valid_img_pre = [os.path.join(data_dir, fname) for fname in valid_df["image_fname_pre"]]
    valid_img_post = [os.path.join(data_dir, fname) for fname in valid_df["image_fname_post"]]
    valid_mask_post = [os.path.join(data_dir, fname) for fname in valid_df["mask_fname_post"]]

    trainset = ImageMaskDataset(
        train_img_pre,
        train_img_post,
        train_mask_post,
        use_edges=use_edges,
        spatial_transform=train_transform,
        color_transform=train_color_augs,
        post_image_transform=post_image_transform,
    )
    validset = ImageMaskDataset(
        valid_img_pre,
        valid_img_post,
        valid_mask_post,
        use_edges=use_edges,
        spatial_transform=valid_transform,
        align_post=align_post
    )

    train_sampler = None
    if balance:
        # destroyed_buildings, destroyed_pixels, event_name, event_type, folder, image_fname, image_id, , light_damaged_pixels, mask_fname, , medium_damaged_pixels, , non_damaged_pixels, sample_id, fold

        non_damaged_buildings = train_df["non_damaged_buildings_post"].values > 0
        light_damaged_buildings = train_df["light_damaged_buildings_post"].values > 0
        medium_damaged_buildings = train_df["medium_damaged_buildings_post"].values > 0
        destroyed_buildings = train_df["destroyed_buildings_post"].values > 0
        labels = (
            non_damaged_buildings * 1
            + light_damaged_buildings * 2
            + medium_damaged_buildings * 4
            + destroyed_buildings * 8
        )

        num_samples = 4 * min(
            sum(non_damaged_buildings),
            sum(light_damaged_buildings),
            sum(medium_damaged_buildings),
            sum(destroyed_buildings),
        )
        weights = compute_sample_weight("balanced", labels)
        train_sampler = WeightedRandomSampler(weights, int(num_samples), replacement=bool(num_samples > len(train_df)))
        print("Using balancing for training", num_samples, weights.min(), weights.mean(), weights.max())
    elif train_on_crops:
        # If we're training on crops, make 3 crops for each sample
        trainset = ConcatDataset([trainset] * int(crops_multiplication_factor))

    if sanity_check:
        first_batch = [trainset[i] for i in range(32)]
        return first_batch * 50, first_batch, None

    return trainset, validset, train_sampler


def make_dual_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_rename = [
        "destroyed_buildings",
        "destroyed_pixels",
        "event_name",
        "event_type",
        "folder",
        "image_fname",
        "image_id",
        "light_damaged_buildings",
        "light_damaged_pixels",
        "mask_fname",
        "medium_damaged_buildings",
        "medium_damaged_pixels",
        "non_damaged_buildings",
        "non_damaged_pixels",
        "sample_id",
        # "fold"
    ]

    df = df.sort_values(by=["image_id"])
    df_pre = df[df["event_type"] == "pre"].copy().reset_index(drop=True)
    df_post = df[df["event_type"] == "post"].copy().reset_index(drop=True)

    df = df_pre.merge(df_post, left_index=True, right_index=True, suffixes=["_pre", "_post"])
    return df


def get_test_dataset(data_dir: str, image_size=(224, 224), use_edges=False, fast=False, align_post=False):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_edges: If True, adds 'edge' target mask
    :param augmentation: Type of image augmentations to use
    :param train_mode:
    'random' - crops tiles from source images randomly.
    'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """
    resize = A.Resize(image_size[0], image_size[1])
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
    )

    valid_transform = A.Compose([resize, normalize])

    test_images_post = [
        fname
        for fname in fs.find_images_in_dir(os.path.join(data_dir, "test", "images"))
        if "post_" in fs.id_from_fname(fname)
    ]

    test_images_pre = [fname.replace("_post_", "_pre_") for fname in test_images_post]

    if fast:
        test_images_pre = test_images_pre[:128]
        test_images_post = test_images_post[:128]

    validset = ImageMaskDataset(
        test_images_pre,
        test_images_post,
        None,
        use_edges=use_edges,
        spatial_transform=valid_transform,
        align_post=align_post,
    )

    return validset


def get_pseudolabeling_dataset(
    data_dir,
    image_size,
    include_masks,
    augmentation,
    use_edges=False,
    train_on_crops=False,
    enable_post_image_transform=False,
    pseudolabels_dir=None
):

    train_transform, train_color_augs, valid_transform, post_image_transform = get_transforms(
        image_size=image_size,
        augmentation=augmentation,
        train_on_crops=train_on_crops,
        enable_post_image_transform=enable_post_image_transform,
    )

    images_dir = os.path.join(data_dir, "test", "images")
    masks_dir = pseudolabels_dir or os.path.join(data_dir, "test", "masks")
    os.makedirs(masks_dir, exist_ok=True)

    test_images_post = [fname for fname in fs.find_images_in_dir(images_dir) if "_post_" in fs.id_from_fname(fname)]
    test_images_pre = [fname.replace("_post_", "_pre_") for fname in test_images_post]

    if include_masks:
        test_masks_post = [os.path.join(masks_dir, os.path.basename(fname)) for fname in test_images_post]
    else:
        test_masks_post = None

    validset = ImageMaskDataset(
        test_images_pre,
        test_images_post,
        test_masks_post,
        use_edges=use_edges,
        color_transform=train_color_augs,
        spatial_transform=train_transform,
        post_image_transform=post_image_transform,
    )

    return validset


def get_classification_datasets(
    data_dir: str,
    min_size=64,
    image_size=(224, 224),
    augmentation="safe",
    sanity_check=False,
    fast=False,
    fold=0,
    enable_post_image_transform=False,
):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_edges: If True, adds 'edge' target mask
    :param augmentation: Type of image augmentations to use
    :param train_mode:
    'random' - crops tiles from source images randomly.
    'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """

    resize_op = A.Compose(
        [
            A.LongestMaxSize(max(image_size[0], image_size[1])),
            A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        ]
    )

    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
    )

    df = pd.read_csv(os.path.join(data_dir, "train_crops.csv"))

    post_transform = None
    if augmentation == "safe":
        augment = A.Compose([A.RandomRotate90(), A.Transpose()])
    elif augmentation == "light":
        augment = A.Compose([A.RandomRotate90(), A.Transpose()])
    else:
        print("Unsupported augmentation", augmentation)
        augment = A.NoOp()

    train_transform = A.Compose([resize_op, augment, normalize])

    valid_transform = A.Compose([resize_op, normalize])

    train_sampler = None

    df = df[df["max_size"] >= min_size]

    train_df = df[df["fold"] != fold]
    valid_df = df[df["fold"] == fold]

    if fast:
        train_df = train_df[:128]
        valid_df = valid_df[:128]
        train_sampler = None

    train_img_pre = [os.path.join(data_dir, "crops", fname) for fname in train_df["pre_crop_fname"]]
    train_img_post = [os.path.join(data_dir, "crops", fname) for fname in train_df["post_crop"]]
    train_targets = np.array(train_df["label"]) - 1  # Targets in CSV starting from 1

    valid_img_pre = [os.path.join(data_dir, "crops", fname) for fname in valid_df["pre_crop_fname"]]
    valid_img_post = [os.path.join(data_dir, "crops", fname) for fname in valid_df["post_crop"]]
    valid_targets = np.array(valid_df["label"]) - 1  # Targets in CSV starting from 1

    print(
        "Sample weights (train,val)",
        compute_class_weight("balanced", np.arange(len(DAMAGE_TYPES)), train_targets),
        compute_class_weight("balanced", np.arange(len(DAMAGE_TYPES)), valid_targets),
    )

    trainset = ImageLabelDataset(
        train_img_pre,
        train_img_post,
        train_targets,
        spatial_transform=train_transform,
        color_transform=None,
        post_image_transform=None,
    )
    validset = ImageLabelDataset(valid_img_pre, valid_img_post, valid_targets, spatial_transform=valid_transform)

    if sanity_check:
        first_batch = [trainset[i] for i in range(32)]
        return first_batch * 50, first_batch, None

    return trainset, validset, train_sampler
