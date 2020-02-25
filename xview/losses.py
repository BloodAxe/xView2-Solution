import math

import torch
from pytorch_toolbelt.losses import *
import torch.nn.functional as F

__all__ = ["get_loss", "AdaptiveMaskLoss2d"]

from torch import nn
from torch.nn import Module, Parameter

from .dataset import UNLABELED_SAMPLE
from .ssim_loss import SSIM
from .utils.inference_image_output import resize_mask_one_hot


class LabelSmoothingCrossEntropy2d(Module):
    """

    Original implementation: fast.ai
    """

    def __init__(self, eps: float = 0.1, reduction="mean", weight=None, ignore_index=-100):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(self, output, target):
        num_classes = output.size(1)
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1)
            if self.reduction == "mean":
                loss = loss.mean()

        return loss * self.eps / num_classes + (1 - self.eps) * F.nll_loss(
            log_preds, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )


class OHEMCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Online hard example mining CE loss

    https://arxiv.org/pdf/1812.05802.pdf
    """

    def __init__(self, weight=None, fraction=0.3, ignore_index=-100, reduction="mean"):
        super().__init__(weight, ignore_index=ignore_index, reduction=reduction)
        self.fraction = fraction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)

        with torch.no_grad():
            positive_mask = (target > 0).view(batch_size, -1)
            Cp = torch.sum(positive_mask, dim=1)  # Number of positive pixels
            Cn = torch.sum(~positive_mask, dim=1)  # Number of negative pixels
            Chn = torch.max((Cn / 4).clamp_min(5), 2 * Cp)

        losses = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        ).view(target.size(0), -1)

        loss = 0
        num_samples = 0

        for i in range(batch_size):
            positive_losses = losses[i, positive_mask[i]]
            negative_losses = losses[i, ~positive_mask[i]]

            num_negatives = Chn[i]
            hard_negative_losses, _ = negative_losses.sort(descending=True)[:num_negatives]

            loss = positive_losses.sum() + hard_negative_losses.sum() + loss

            num_samples += positive_losses.size(0)
            num_samples += hard_negative_losses.size(0)

        loss /= float(num_samples)
        return loss


def get_loss(loss_name: str, ignore_index=UNLABELED_SAMPLE):
    if loss_name.lower() == "bce":
        return BCELoss(ignore_index=ignore_index)

    if loss_name.lower() == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    if loss_name.lower() == "ohem_ce":
        return OHEMCrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0])).cuda()

    if loss_name.lower() == "weighted_ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0])).cuda()

    if loss_name.lower() == "weighted2_ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor([1.0, 1.0, 3.0, 2.0, 1.0])).cuda()

    if loss_name.lower() == "dsv_ce":
        return AdaptiveMaskLoss2d(
            nn.CrossEntropyLoss(ignore_index=ignore_index, weight=torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0]))
        ).cuda()

    if loss_name.lower() in {"ce_building_only", "ce_buildings_only"}:
        # This ignores predictions on "non-building" pixels
        return nn.CrossEntropyLoss(ignore_index=0)

    if loss_name.lower() == "soft_bce":
        return SoftBCELoss(smooth_factor=0.1, ignore_index=ignore_index)

    if loss_name.lower() == "soft_ce":
        return LabelSmoothingCrossEntropy2d(eps=0.1, ignore_index=ignore_index)

    if loss_name.lower() == "binary_focal":
        return BinaryFocalLoss(alpha=None, gamma=2, ignore_index=ignore_index)

    if loss_name.lower() == "focal":
        return FocalLoss(alpha=None, gamma=2, ignore_index=ignore_index, reduction="mean")

    if loss_name.lower() == "nfl":
        return FocalLoss(alpha=None, gamma=2, ignore_index=ignore_index, normalized=True, reduction="sum")

    if loss_name.lower() == "dice":
        return DiceLoss(mode="multiclass")

    if loss_name.lower() == "log_dice":
        return DiceLoss(mode="multiclass", log_loss=True)

    if loss_name.lower() == "am-softmax":
        return AmSoftmax2d(weight=torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0])).cuda()

    if loss_name.lower() == "arcface":
        return ArcFaceLoss2d(ignore_index=ignore_index)

    if loss_name.lower() == "ssim":
        return SSIM(5).cuda()

    raise KeyError(loss_name)


class AdaptiveMaskLoss2d(nn.Module):
    """
    Works only with sigmoid masks and bce loss
    Rescales target mask to predicted mask
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            target_one_hot = F.one_hot(target, int(input.size(1))).permute(0, 3, 1, 2).type(input.dtype)

            scale = int(target.size(2)) // int(input.size(2))
            while scale > 2:
                target_one_hot = F.interpolate(target_one_hot, scale_factor=0.5, mode="bilinear", align_corners=False)
                scale = int(target_one_hot.size(2)) // int(input.size(2))

            target_one_hot = F.interpolate(target_one_hot, size=input.size()[2:], mode="bilinear", align_corners=False)

        target = target_one_hot.argmax(dim=1).type(target.dtype)
        return self.loss(input, target)


class ArcFaceLoss2d(nn.modules.Module):
    """
    https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109#latest-560973
    """

    def __init__(self, s=30.0, m=0.35, gamma=1, ignore_index=-100):
        super(ArcFaceLoss2d, self).__init__()
        self.gamma = gamma
        self.classify_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.s = s
        self.easy_margin = False
        self.cos_m = float(math.cos(m))
        self.sin_m = float(math.sin(m))
        self.th = float(math.cos(math.pi - m))
        self.mm = float(math.sin(math.pi - m) * m)

    def forward(self, cos_theta: torch.Tensor, labels):
        num_classes = cos_theta.size(1)
        sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        phi = (cos_theta * self.cos_m - sine * self.sin_m).type(cos_theta.dtype)
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = F.one_hot(labels, num_classes).type(cos_theta.dtype)
        one_hot = one_hot.permute(0, 3, 1, 2)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cos_theta, labels)

        loss = (loss1 + self.gamma * loss2) / (1 + self.gamma)
        return loss


class AmSoftmax2d(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, ignore_index=UNLABELED_SAMPLE, weight=None):
        super(AmSoftmax2d, self).__init__()
        # initial kernel

        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.0  # see normface https://arxiv.org/abs/1704.06369
        self.classify_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)

    def forward(self, cos_theta, labels):

        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m

        num_classes = cos_theta.size(1)

        one_hot = F.one_hot(labels, num_classes)  # .type(embbedings.dtype)
        one_hot = one_hot.permute(0, 3, 1, 2)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return self.classify_loss(output, labels)
