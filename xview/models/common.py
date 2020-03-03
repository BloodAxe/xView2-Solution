from pytorch_toolbelt.modules import GlobalAvgPool2d, ABN, GlobalMaxPool2d
from torch import nn


def disaster_type_classifier(features: int, num_classes: int, embedding=256, abn_block=ABN, dropout=0.0) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(features, embedding, kernel_size=1),
        abn_block(embedding),
        GlobalAvgPool2d(flatten=True),
        nn.Dropout(dropout, inplace=True),
        nn.Linear(embedding, num_classes),
    )


def damage_types_classifier(features: int, num_classes: int, embedding=256, abn_block=ABN, dropout=0.0) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(features, embedding, kernel_size=1),
        abn_block(embedding),
        nn.Dropout2d(dropout),
        nn.Conv2d(embedding, num_classes, kernel_size=1),
        GlobalMaxPool2d(flatten=True),
    )
