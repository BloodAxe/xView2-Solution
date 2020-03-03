from functools import partial
from typing import List, Union, Callable

import torch
from pytorch_toolbelt.modules import ABN, ACT_RELU, ACT_SWISH
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import DecoderModule
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from .common import disaster_type_classifier, damage_types_classifier
from ..dataset import OUTPUT_MASK_KEY, DISASTER_TYPE_KEY, DISASTER_TYPES, DAMAGE_TYPE_KEY, DAMAGE_TYPES

__all__ = ["UnetV2SegmentationModel"]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UNetDecoderV2(DecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: List[int],
        mask_channels: int,
        last_upsample_filters=None,
        dropout=0.0,
        abn_block=ABN,
    ):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]

        if last_upsample_filters is None:
            last_upsample_filters = decoder_features[0]

        self.encoder_features = feature_maps
        self.decoder_features = decoder_features
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_features))])

        self.bottlenecks = nn.ModuleList(
            [
                ConvBottleneck(self.encoder_features[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_features[:]))
            ]
        )

        self.output_filters = decoder_features

        self.last_upsample = UnetDecoderBlock(decoder_features[0], last_upsample_filters, last_upsample_filters)

        self.final = nn.Conv2d(last_upsample_filters, mask_channels, kernel_size=1)

    def get_decoder(self, layer):
        in_channels = (
            self.encoder_features[layer + 1]
            if layer + 1 == len(self.decoder_features)
            else self.decoder_features[layer + 1]
        )
        return UnetDecoderBlock(in_channels, self.decoder_features[layer], self.decoder_features[max(layer, 0)])

    def forward(self, feature_maps):

        last_dec_out = feature_maps[-1]

        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            decoder = self.decoder_stages[rev_idx]
            x = decoder(x)
            x = bottleneck(x, feature_maps[rev_idx - 1])

        x = self.last_upsample(x)

        f = self.final(x)

        return f


class UnetV2SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        disaster_type_classes: int,
        damage_type_classes: int,
        unet_channels: List[int],
        dropout=0.25,
        abn_block: Union[ABN, Callable[[int], nn.Module]] = ABN,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        feature_maps = [2 * fm for fm in encoder.output_filters]

        self.decoder = UNetDecoderV2(
            feature_maps=feature_maps,
            decoder_features=unet_channels,
            mask_channels=num_classes,
            dropout=dropout,
            abn_block=abn_block,
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


def efficientb3_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.EfficientNetB3Encoder(pretrained=pretrained,
                                      layers=[0, 1, 2, 4, 6],
                                      abn_params={"activation": ACT_RELU})
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[64, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet121_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[64, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet169_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.DenseNet169Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[128, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )

def resnet18_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet18Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[64, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )
def resnet34_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[64, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def resnet50_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[96, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def resnet101_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.Resnet101Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[64, 128, 256, 384],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def seresnext50_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[64, 128, 256, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def seresnext101_unet_v2(input_channels=6, num_classes=5, dropout=0.0, pretrained=True, classifiers=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    return UnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES) if classifiers else None,
        damage_type_classes=len(DAMAGE_TYPES) if classifiers else None,
        unet_channels=[128, 128, 256, 384],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )
