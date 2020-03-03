from functools import partial
from typing import Optional

import torch
from pytorch_toolbelt.modules import ABN, ACT_RELU
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import FPNCatDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from .common import disaster_type_classifier, damage_types_classifier
from ..dataset import OUTPUT_MASK_KEY, DISASTER_TYPE_KEY, DISASTER_TYPES, DAMAGE_TYPE_KEY, DAMAGE_TYPES


class FPNSumFinalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(input_channels)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(input_channels)

        self.final = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.final(x)
        return x


class FPNCatFinalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(input_channels)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        return x


class FPNCatSegmentationModelV2(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        disaster_type_classes: int,
        damage_type_classes: int,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        full_size_mask=True,
        interpolation_mode: str = "bilinear",
        align_corners: Optional[bool] = False,
    ):
        super().__init__()
        self.encoder = encoder

        feature_maps = [2 * fm for fm in encoder.output_filters]

        self.decoder = FPNCatDecoder(
            feature_maps=feature_maps,
            output_channels=num_classes,
            dsv_channels=None,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
            final_block=partial(FPNCatFinalBlock, abn_block=abn_block),
            interpolation_mode=interpolation_mode,
            align_corners=align_corners,
        )

        self.full_size_mask = full_size_mask
        if disaster_type_classes is not None:
            self.disaster_type_classifier = disaster_type_classifier(
                feature_maps[-1], disaster_type_classes, dropout=dropout
            )
        else:
            self.disaster_type_classifier = None

        if damage_type_classes is not None:
            self.damage_types_classifier = damage_types_classifier(
                feature_maps[-1], damage_type_classes, dropout=dropout
            )
        else:
            self.damage_types_classifier = None

    def forward(self, x):
        batch_size = x.size(0)
        pre, post = x[:, 0:3, ...], x[:, 3:6, ...]

        if self.training:
            x = torch.cat([pre, post], dim=0)
            features = self.encoder(x)
            features = [torch.cat([f[0:batch_size], f[batch_size : batch_size * 2]], dim=1) for f in features]
        else:
            pre_features, post_features = self.encoder(pre), self.encoder(post)
            features = [torch.cat([pre, post], dim=1) for pre, post in zip(pre_features, post_features)]

        # Decode mask
        mask = self.decoder(features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.disaster_type_classifier is not None:
            disaster_type = self.disaster_type_classifier(features[-1])
            output[DISASTER_TYPE_KEY] = disaster_type

        if self.damage_types_classifier is not None:
            damage_types = self.damage_types_classifier(features[-1])
            output[DAMAGE_TYPE_KEY] = damage_types

        return output


def resnet34_fpncatv2_256_nearest(num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    return FPNCatSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        fpn_channels=256,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
        interpolation_mode="nearest",
        align_corners=None,
    )


def resnet34_fpncatv2_256(num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    return FPNCatSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        fpn_channels=256,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def resnet101_fpncatv2_256(num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet101Encoder(pretrained=pretrained)
    return FPNCatSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        fpn_channels=256,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet201_fpncatv2_256(num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.DenseNet201Encoder(pretrained=pretrained)
    return FPNCatSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        fpn_channels=256,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def inceptionv4_fpncatv2_256(num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.InceptionV4Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    return FPNCatSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        fpn_channels=256,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def efficientb4_fpncatv2_256(num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.EfficientNetB4Encoder()
    return FPNCatSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        fpn_channels=256,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )
