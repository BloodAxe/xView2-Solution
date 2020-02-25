import torch
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn

from xview.dataset import OUTPUT_MASK_KEY
from xview.losses import ArcFaceLoss2d, OHEMCrossEntropyLoss
from xview.models.deeplab import resnet34_deeplab128
from xview.models.fpn_v2 import (
    resnet101_fpncatv2_256,
    densenet201_fpncatv2_256,
    efficientb4_fpncatv2_256,
    inceptionv4_fpncatv2_256,
)
from xview.models.hrnet_arc import hrnet18_arc
from xview.models.segcaps import SegCaps
from xview.models.unet import resnet18_unet32
from xview.models.unetv2 import inceptionv4_unet_v2, resnet101_unet_v2


def test_ohem_ce():
    x = torch.randn((8, 5, 128, 128)).cuda()
    y = torch.randint(0, 5, (8, 128, 128)).long().cuda()

    loss = OHEMCrossEntropyLoss()
    l = loss(x, y)
    print(l)


def test_conv_transpose():
    x = torch.randn((1, 32, 128, 128)).cuda()

    module = nn.ConvTranspose2d(32, 5, kernel_size=8, stride=4, padding=2).cuda()

    y = module(x)
    print(y.size())


@torch.no_grad()
def test_hrnet18_arc():
    x = torch.randn((1, 6, 256, 256))
    net = hrnet18_arc().eval()

    out = net(x)
    tgt = torch.randint(0, 5, (1, 256, 256)).long()
    criterion = ArcFaceLoss2d()
    loss = criterion(out[OUTPUT_MASK_KEY], tgt)

    print(out)


@torch.no_grad()
def test_resnet18_unet():
    x = torch.randn((1, 6, 256, 256))
    net = resnet18_unet32().eval()
    print(count_parameters(net))
    out = net(x)
    print(out)


@torch.no_grad()
def test_resnet34_deeplab128():
    x = torch.randn((1, 6, 512, 512))
    net = resnet34_deeplab128().eval()
    print(count_parameters(net))
    out = net(x)
    print(out)


def test_seg_caps():
    net = SegCaps(num_classes=5)
    print(count_parameters(net))
    x = torch.randn((4, 3, 256, 256))
    y = net(x)
    print(y.size())


def test_selim_unet():
    from xview.models.selim.unet import DensenetUnet

    d = DensenetUnet(5, backbone_arch="densenet121")
    d.eval()
    import numpy as np

    with torch.no_grad():
        images = torch.from_numpy(np.zeros((16, 3, 256, 256), dtype="float32"))
        i = d(images)
        print(i.shape)

    print(d)


@torch.no_grad()
def test_inception_unet_like_selim():
    d = inceptionv4_unet_v2().cuda().eval()
    print(count_parameters(d))

    print(d.decoder.decoder_features)
    print(d.decoder.bottlenecks)
    print(d.decoder.decoder_stages)

    images = torch.rand(4, 6, 512, 512).cuda()
    i = d(images)
    print(i[OUTPUT_MASK_KEY].size())


@torch.no_grad()
def test_inception_unet_like_selim():
    d = resnet101_unet_v2().cuda().eval()
    print(count_parameters(d))

    print(d.decoder.decoder_features)
    print(d.decoder.bottlenecks)
    print(d.decoder.decoder_stages)

    images = torch.rand(4, 6, 512, 512).cuda()
    i = d(images)
    print(i[OUTPUT_MASK_KEY].size())


@torch.no_grad()
def test_resnet101_fpncatv2_256():
    d = resnet101_fpncatv2_256().cuda().eval()
    print(count_parameters(d))

    images = torch.rand(2, 6, 512, 512).cuda()
    i = d(images)
    print(i[OUTPUT_MASK_KEY].size())


@torch.no_grad()
def test_densenet201_fpncatv2_256():
    d = densenet201_fpncatv2_256().cuda().eval()
    print(count_parameters(d))

    images = torch.rand(4, 6, 512, 512).cuda()
    i = d(images)
    print(i[OUTPUT_MASK_KEY].size())


@torch.no_grad()
def test_inceptionv4_fpncatv2_256():
    d = inceptionv4_fpncatv2_256().cuda().eval()
    print(count_parameters(d))

    images = torch.rand(2, 6, 512, 512).cuda()
    i = d(images)
    print(i[OUTPUT_MASK_KEY].size())


@torch.no_grad()
def test_efficientb4_fpncatv2_256():
    d = efficientb4_fpncatv2_256().cuda().eval()
    print(count_parameters(d))

    images = torch.rand(4, 6, 512, 512).cuda()
    i = d(images)
    print(i[OUTPUT_MASK_KEY].size())
