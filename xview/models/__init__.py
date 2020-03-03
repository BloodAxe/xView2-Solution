from torch import nn

from . import fpn, unet, hrnet, unetv2, cls, fpn_v2, hrnet_v2, fpn_v3, unetv3

__all__ = ["get_model"]


def get_model(model_name: str, dropout=0.0, pretrained=True, classifiers=True) -> nn.Module:
    registry = {
        # FPN V2
        "resnet34_fpncatv2_256": fpn_v2.resnet34_fpncatv2_256,
        "resnet34_fpncatv2_256_nearest": fpn_v2.resnet34_fpncatv2_256_nearest,
        "densenet201_fpncatv2_256": fpn_v2.densenet201_fpncatv2_256,
        "resnet101_fpncatv2_256": fpn_v2.resnet101_fpncatv2_256,
        "efficientb4_fpncatv2_256": fpn_v2.efficientb4_fpncatv2_256,
        "inceptionv4_fpncatv2_256": fpn_v2.inceptionv4_fpncatv2_256,

        # UNet
        "resnet18_unet_v2": unetv2.resnet18_unet_v2,
        "resnet34_unet_v2": unetv2.resnet34_unet_v2,
        "resnet50_unet_v2": unetv2.resnet50_unet_v2,
        "resnet101_unet_v2": unetv2.resnet101_unet_v2,
        "seresnext50_unet_v2": unetv2.seresnext50_unet_v2,
        "seresnext101_unet_v2": unetv2.seresnext101_unet_v2,
        "densenet121_unet_v2": unetv2.densenet121_unet_v2,
        "densenet169_unet_v2": unetv2.densenet169_unet_v2,
        "efficientb3_unet_v2": unetv2.efficientb3_unet_v2,


    }

    return registry[model_name.lower()](dropout=dropout, pretrained=pretrained, classifiers=classifiers)
