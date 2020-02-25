import os

import torch
from typing import Optional, Dict, List, Tuple

from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt.utils import fs
from torch.nn import functional as F
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_toolbelt.inference.functional as AF

from xview.dataset import (
    OUTPUT_MASK_POST_KEY,
    OUTPUT_MASK_PRE_KEY,
    INPUT_IMAGE_ID_KEY,
    INPUT_IMAGE_PRE_KEY,
    INPUT_IMAGE_POST_KEY,
    OUTPUT_MASK_KEY,
    INPUT_IMAGE_KEY,
    DAMAGE_TYPE_KEY,
    INPUT_MASK_KEY,
)
from xview.metric import CompetitionMetricCallback
from xview.models import get_model
from xview.postprocessing import (
    make_predictions_dominant,
    make_predictions_naive,
    make_predictions_floodfill,
    make_predictions_dominant_v2,
    make_pseudolabeling_target,
)
from xview.train_utils import report_checkpoint
from xview.utils.inference_image_output import colorize_mask

import numpy as np


class ApplySigmoidTo(nn.Module):
    def __init__(self, model, input_key="logits"):
        super().__init__()
        self.model = model
        self.input_key = input_key

    def forward(self, *input, **kwargs) -> Dict:
        output = self.model(*input, **kwargs)
        if self.input_key in output:
            output[self.input_key] = output[self.input_key].sigmoid()
        return output


class ApplySoftmaxTo(nn.Module):
    def __init__(self, model, input_key="logits"):
        super().__init__()
        self.model = model
        self.input_key = input_key

    def forward(self, *input, **kwargs) -> Dict:
        output = self.model(*input, **kwargs)
        if self.input_key in output:
            output[self.input_key] = output[self.input_key].softmax(dim=1)
        return output


class HFlipTTA(nn.Module):
    def __init__(self, model, outputs, average=True):
        super().__init__()
        self.model = model
        self.outputs = outputs
        self.average = average

    def forward(self, image):
        outputs = self.model(image)
        outputs_flip = self.model(AF.torch_fliplr(image))

        for output_key in self.outputs:
            outputs[output_key] += AF.torch_fliplr(outputs_flip[output_key])

        if self.average:
            averaging_scale = 0.5
            for output_key in self.outputs:
                outputs[output_key] *= averaging_scale

        return outputs


class D4TTA(nn.Module):
    def __init__(self, model, outputs, average=True):
        super().__init__()
        self.model = model
        self.outputs = outputs
        self.average = average

    def forward(self, image):
        outputs = self.model(image)

        augment = [AF.torch_rot90, AF.torch_rot180, AF.torch_rot270]
        deaugment = [AF.torch_rot270, AF.torch_rot180, AF.torch_rot90]

        for aug, deaug in zip(augment, deaugment):
            input = aug(image)
            aug_output = self.model(input)

            for output_key in self.outputs:
                outputs[output_key] += deaug(aug_output[output_key])

        image_t = AF.torch_transpose(image)

        augment = [AF.torch_none, AF.torch_rot90, AF.torch_rot180, AF.torch_rot270]
        deaugment = [AF.torch_none, AF.torch_rot270, AF.torch_rot180, AF.torch_rot90]

        for aug, deaug in zip(augment, deaugment):
            input = aug(image_t)
            aug_output = self.model(input)

            for output_key in self.outputs:
                x = deaug(aug_output[output_key])
                outputs[output_key] += AF.torch_transpose(x)

        if self.average:
            averaging_scale = 1.0 / 8.0
            for output_key in self.outputs:
                outputs[output_key] *= averaging_scale

        return outputs


class MultiscaleTTA(nn.Module):
    def __init__(self, model, outputs, size_offsets: List[int], average=True):
        super().__init__()
        self.model = model
        self.outputs = outputs
        self.size_offsets = size_offsets
        self.average = average

    def integrate(self, outputs, input, augment, deaugment):
        aug_input = augment(input)
        aug_output = self.model(aug_input)

        for output_key in self.outputs:
            outputs[output_key] += deaugment(aug_output[output_key])

    def forward(self, image):
        outputs = self.model(image)
        x_size_orig = image.size()[2:]

        for image_size_offset in self.size_offsets:
            x_size_modified = x_size_orig[0] + image_size_offset, x_size_orig[1] + image_size_offset
            self.integrate(
                outputs,
                image,
                lambda x: F.interpolate(x, size=x_size_modified, mode="bilinear", align_corners=False),
                lambda x: F.interpolate(x, size=x_size_orig, mode="bilinear", align_corners=False),
            )

        if self.average:
            averaging_scale = 1.0 / (len(self.size_offsets) + 1)
            for output_key in self.outputs:
                outputs[output_key] *= averaging_scale
        return outputs


class Ensembler(nn.Module):
    def __init__(self, models: List[nn.Module], outputs: List[str]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.outputs = outputs

    def forward(self, *input, **kwargs):
        num_models = len(self.models)

        with tqdm(total=num_models, desc="Inference") as tq:
            output_0 = self.models[0](*input, **kwargs)
            tq.update()

            for i in range(1, num_models):
                output_i = self.models[i](*input, **kwargs)
                tq.update()

                # Aggregate predictions
                for key in self.outputs:
                    output_0[key] += output_i[key]

        scale = 1.0 / num_models
        return {key: output_0[key] * scale for key in self.outputs}


class ApplyWeights(nn.Module):
    def __init__(self, model, weights, output_key=OUTPUT_MASK_KEY):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights).float().view(1, -1, 1, 1)

        super().__init__()
        self.model = model
        self.register_buffer("weights", weights)
        self.output_key = output_key

    def forward(self, x):
        output = self.model(x)
        output[self.output_key] *= self.weights
        return output


def model_from_checkpoint(
    model_checkpoint: str, tta: Optional[str] = None, activation_after="model", model=None, report=True, classifiers=True
) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model_name = model or checkpoint["checkpoint_data"]["cmd_args"]["model"]

    score = float(checkpoint["epoch_metrics"]["valid"]["weighted_f1"])
    loc = float(checkpoint["epoch_metrics"]["valid"]["weighted_f1/localization_f1"])
    dmg = float(checkpoint["epoch_metrics"]["valid"]["weighted_f1/damage_f1"])
    fold = int(checkpoint["checkpoint_data"]["cmd_args"]["fold"])

    if report:
        print(model_checkpoint, model_name)
        report_checkpoint(checkpoint)

    model = get_model(model_name, pretrained=False, classifiers=classifiers)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    del checkpoint

    if activation_after == "model":
        model = ApplySoftmaxTo(model, OUTPUT_MASK_KEY)

    if tta == "multiscale":
        print(f"Using {tta}")
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-256, -128, +128, +256], average=True)

    if tta == "flip":
        print(f"Using {tta}")
        model = HFlipTTA(model, outputs=[OUTPUT_MASK_KEY], average=True)

    if tta == "flipscale":
        print(f"Using {tta}")
        model = HFlipTTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-256, -128, +128, +256], average=True)

    if tta == "multiscale_d4":
        print(f"Using {tta}")
        model = D4TTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-256, -128, +128, +256], average=True)

    if activation_after == "tta":
        model = ApplySoftmaxTo(model, OUTPUT_MASK_KEY)

    info = {
        "model": fs.id_from_fname(model_checkpoint),
        "model_name": model_name,
        "fold": fold,
        "score": score,
        "localization": loc,
        "damage": dmg,
    }
    return model, info


@torch.no_grad()
def run_inference_on_dataset(
    model, dataset, output_dir, batch_size=1, workers=0, weights=None, fp16=False, cpu=False, postprocessing="naive", save_pseudolabels=True
):
    if not cpu:
        if fp16:
            model = model.half()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print("Using multi-GPU inference")

    model = model.eval()

    if weights is not None:
        print("Using weights", weights)
        weights = torch.tensor(weights).float().view(1, -1, 1, 1)

        if not cpu:
            if fp16:
                weights = weights.half()
            weights = weights.cuda()

    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=not cpu, num_workers=workers)

    pseudolabeling_dir = os.path.join(output_dir + "_pseudolabeling")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pseudolabeling_dir, exist_ok=True)

    postprocessings = {}
    if postprocessing == "naive":
        postprocessings[postprocessing] = make_predictions_naive
    elif postprocessing == "dominant":
        postprocessings[postprocessing] = make_predictions_dominant
    elif postprocessing in {"dominant2", "dominantv2", "dominant_v2"}:
        postprocessings[postprocessing] = make_predictions_dominant_v2
    elif postprocessing == "floodfill":
        postprocessings[postprocessing] = make_predictions_floodfill
    elif postprocessing is None:
        postprocessings = {
            "naive": make_predictions_naive,
            "dominant": make_predictions_dominant,
            "dominantv2": make_predictions_dominant_v2,
            "floodfill": make_predictions_floodfill,
        }

    for batch in tqdm(data_loader):
        image = batch[INPUT_IMAGE_KEY]

        if not cpu:
            if fp16:
                image = image.half()
            image = image.cuda(non_blocking=True)

        image_ids = batch[INPUT_IMAGE_ID_KEY]

        output = model(image)

        masks = output[OUTPUT_MASK_KEY]

        if weights is not None:
            masks *= weights

        if masks.size(2) != 1024 or masks.size(3) != 1024:
            masks = F.interpolate(masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks = to_numpy(masks).astype(np.float32)

        for i, image_id in enumerate(image_ids):
            _, _, image_uuid = image_id.split("_")

            # Save pseudolabeling target
            if save_pseudolabels:
                pseudo_mask = make_pseudolabeling_target(masks[i])
                pseudo_mask = pseudo_mask.astype(np.uint8)
                pseudo_mask = colorize_mask(pseudo_mask)
                pseudo_mask.save(os.path.join(pseudolabeling_dir, f"test_post_{image_uuid}.png"))

            for postprocessing_name, postprocessing_fn in postprocessings.items():

                output_dir_for_postprocessing = os.path.join(output_dir + "_" + postprocessing_name)
                os.makedirs(output_dir_for_postprocessing, exist_ok=True)

                localization_image, damage_image = postprocessing_fn(masks[i])

                localization_fname = os.path.join(
                    output_dir_for_postprocessing, f"test_localization_{image_uuid}_prediction.png"
                )
                localization_image = colorize_mask(localization_image)
                localization_image.save(localization_fname)

                damage_fname = os.path.join(output_dir_for_postprocessing, f"test_damage_{image_uuid}_prediction.png")
                damage_image = colorize_mask(damage_image)
                damage_image.save(damage_fname)

    del data_loader


@torch.no_grad()
def run_inference_on_dataset_oof(model, dataset, output_dir, batch_size=1, workers=0, save=True, fp16=False):
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.eval()
    if fp16:
        model = model.half()

    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers)

    if save:
        os.makedirs(output_dir, exist_ok=True)

    allrows = []

    for batch in tqdm(data_loader):
        image = batch[INPUT_IMAGE_KEY]
        if fp16:
            image = image.half()

        image = image.cuda(non_blocking=True)
        image_ids = batch[INPUT_IMAGE_ID_KEY]

        dmg_true = to_numpy(batch[INPUT_MASK_KEY]).astype(np.float32)

        output = model(image)

        masks = output[OUTPUT_MASK_KEY]
        masks = to_numpy(masks)

        for i, image_id in enumerate(image_ids):
            damage_mask = masks[i]

            if save:
                damage_fname = os.path.join(output_dir, fs.change_extension(image_id.replace("_pre", "_post"), ".npy"))
                np.save(damage_fname, damage_mask.astype(np.float16))

            loc_pred, dmg_pred = make_predictions_naive(damage_mask)
            row = CompetitionMetricCallback.get_row_pair(loc_pred, dmg_pred, dmg_true[i], dmg_true[i])
            allrows.append(row)

        if save:
            if DAMAGE_TYPE_KEY in output:
                damage_type = to_numpy(output[DAMAGE_TYPE_KEY].sigmoid()).astype(np.float32)

                for i, image_id in enumerate(image_ids):
                    damage_fname = os.path.join(
                        output_dir, fs.change_extension(image_id.replace("_pre", "_damage_type"), ".npy")
                    )
                    np.save(damage_fname, damage_type[i])

    del data_loader

    return CompetitionMetricCallback.compute_metrics(allrows)


@torch.no_grad()
def run_dual_inference_on_dataset(model, dataset, output_dir, batch_size=1, workers=0):
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers)

    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(data_loader):
        image_pre = batch[INPUT_IMAGE_PRE_KEY].cuda(non_blocking=True)
        image_post = batch[INPUT_IMAGE_POST_KEY].cuda(non_blocking=True)
        image_ids = batch[INPUT_IMAGE_ID_KEY]

        output = model(image_pre=image_pre, image_post=image_post)

        masks_pre = output[OUTPUT_MASK_PRE_KEY]
        if masks_pre.size(2) != 1024 or masks_pre.size(3) != 1024:
            masks_pre = F.interpolate(masks_pre, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks_pre = to_numpy(masks_pre.squeeze(1)).astype(np.float32)

        masks_post = output[OUTPUT_MASK_POST_KEY]
        if masks_post.size(2) != 1024 or masks_post.size(3) != 1024:
            masks_post = F.interpolate(masks_post, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks_post = to_numpy(masks_post).astype(np.float32)

        for i, image_id in enumerate(image_ids):
            _, _, image_uuid = image_id.split("_")
            localization_image = masks_pre[i]
            damage_image = masks_post[i]

            localization_fname = os.path.join(output_dir, f"test_localization_{image_uuid}_prediction.png")
            localization_image = (localization_image > 0.5).astype(np.uint8)
            localization_image = colorize_mask(localization_image)
            localization_image.save(localization_fname)

            damage_fname = os.path.join(output_dir, f"test_damage_{image_uuid}_prediction.png")
            damage_image = np.argmax(damage_image, axis=0).astype(np.uint8)
            damage_image = colorize_mask(damage_image)
            damage_image.save(damage_fname)

    del data_loader


@torch.no_grad()
def run_dual_inference_on_dataset_oof(model, dataset, output_dir, batch_size=1, workers=0):
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=workers)

    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(data_loader):
        image_pre = batch[INPUT_IMAGE_PRE_KEY].cuda(non_blocking=True)
        image_post = batch[INPUT_IMAGE_POST_KEY].cuda(non_blocking=True)
        image_ids = batch[INPUT_IMAGE_ID_KEY]

        output = model(image_pre=image_pre, image_post=image_post)

        masks_pre = output[OUTPUT_MASK_PRE_KEY]
        if masks_pre.size(2) != 1024 or masks_pre.size(3) != 1024:
            masks_pre = F.interpolate(masks_pre, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks_pre = to_numpy(masks_pre.squeeze(1)).astype(np.float32)

        masks_post = output[OUTPUT_MASK_POST_KEY]
        if masks_post.size(2) != 1024 or masks_post.size(3) != 1024:
            masks_post = F.interpolate(masks_post, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks_post = to_numpy(masks_post).astype(np.float32)

        for i, image_id in enumerate(image_ids):
            localization_image = masks_pre[i]
            damage_image = masks_post[i]

            localization_fname = os.path.join(output_dir, fs.change_extension(image_id, ".npy"))
            np.save(localization_fname, localization_image)

            damage_fname = os.path.join(output_dir, fs.change_extension(image_id.replace("_pre", "_post"), ".npy"))
            np.save(damage_fname, damage_image)

    del data_loader
