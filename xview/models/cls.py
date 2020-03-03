import torch
from pytorch_toolbelt.modules import GlobalMaxPool2d
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from pytorch_toolbelt.modules import encoders as E

from xview.dataset import OUTPUT_EMBEDDING_KEY, DAMAGE_TYPE_KEY, DAMAGE_TYPES


class DamageTypeClassificationModel(nn.Module):
    def __init__(self, encoder: EncoderModule, damage_type_classes: int, dropout=0.25):
        super().__init__()
        self.encoder = encoder

        features = encoder.output_filters[-1]

        self.max_pool = GlobalMaxPool2d(flatten=True)
        self.damage_types_classifier = nn.Sequential(
            nn.Linear(features * 2, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(features, damage_type_classes),
        )

    def forward(self, x):
        pre, post = x[:, 0:3, ...], x[:, 3:6, ...]
        pre_features = self.max_pool(self.encoder(pre)[-1])
        post_features = self.max_pool(self.encoder(post)[-1])

        x = torch.cat([pre_features, post_features], dim=1)

        # Decode mask

        output = {OUTPUT_EMBEDDING_KEY: x}

        damage_types = self.damage_types_classifier(x)
        output[DAMAGE_TYPE_KEY] = damage_types

        return output

def resnet18_cls(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet18Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    return DamageTypeClassificationModel(encoder, damage_type_classes=len(DAMAGE_TYPES), dropout=dropout)

def resnet34_cls(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    return DamageTypeClassificationModel(encoder, damage_type_classes=len(DAMAGE_TYPES), dropout=dropout)
