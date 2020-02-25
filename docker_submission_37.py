import argparse
import os
import time

import albumentations as A

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

from xview.dataset import OUTPUT_MASK_KEY, read_image
from xview.inference import model_from_checkpoint, ApplyWeights, MultiscaleTTA, HFlipTTA, Ensembler
from xview.postprocessing import make_predictions_naive
from xview.utils.inference_image_output import colorize_mask


def weighted_model(checkpoint_fname: str, weights, activation: str):
    model, info = model_from_checkpoint(
        fs.auto_file(checkpoint_fname, where="ensemble"), activation_after=activation, report=False, classifiers=False
    )
    model = ApplyWeights(model, weights)
    return model, info


@torch.no_grad()
def main():
    start = time.time()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("pre_image", type=str)
    parser.add_argument("post_image", type=str)
    parser.add_argument("loc_image", type=str)
    parser.add_argument("dmg_image", type=str)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--color-mask", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    pre_image = args.pre_image
    post_image = args.post_image
    localization_fname = args.loc_image
    damage_fname = args.dmg_image
    save_raw = args.raw
    color_mask = args.color_mask
    use_gpu = args.gpu

    size = 1024
    postprocess = "naive"
    image_size = size, size

    print("pre_image   ", pre_image)
    print("post_image  ", post_image)
    print("loc_image   ", localization_fname)
    print("dmg_image   ", damage_fname)
    print("Size        ", image_size)
    print("Postprocess ", postprocess)
    print("Colorize    ", color_mask)
    raw_predictions_file = fs.change_extension(damage_fname, ".npy")
    print("raw_predictions_file", raw_predictions_file)
    print(*torch.__config__.show().split("\n"), sep="\n")

    if not os.path.isdir(os.path.dirname(localization_fname)):
        print("Output directory does not exists", localization_fname)
        return -1

    if not os.access(os.path.dirname(localization_fname), os.W_OK):
        print("Output directory does not have write access", localization_fname)
        return -2

    if not os.path.isdir(os.path.dirname(damage_fname)):
        print("Output directory does not exists", damage_fname)
        return -1

    if not os.access(os.path.dirname(damage_fname), os.W_OK):
        print("Output directory does not have write access", damage_fname)
        return -2

    fold_0_models_dict = [
        # (
        #     "Dec15_21_41_resnet101_fpncatv2_256_512_fold0_fp16_crops.pth",
        #     [0.45136154, 1.4482629, 1.42098208, 0.6839698, 0.96800456],
        # ),
        # (
        #     "Dec16_08_26_resnet34_unet_v2_512_fold0_fp16_crops.pth",
        #     [0.92919105, 1.03831743, 1.03017048, 0.98257118, 1.0241164],
        # ),
        # (
        #     "Dec21_21_54_densenet161_deeplab256_512_fold0_fp16_crops.pth",
        #     [0.48157651, 1.02084685, 1.36264406, 1.03175205, 1.11758873],
        # ),
        # 0.762814651939279 0.854002889559006 0.7237339786736817 [0.9186602573598759, 0.5420118318644089, 0.7123870673168781, 0.8405837378060299] coeffs [0.51244243 1.42747062 1.23648384 0.90290896 0.88912514]
        (
            "Dec30_15_34_resnet34_unet_v2_512_fold0_fp16_pseudo_crops.pth",
            [0.51244243, 1.42747062, 1.23648384, 0.90290896, 0.88912514],
        ),
        # 0.7673669954814148 0.8582940771677703 0.7283982461872626 [0.919932857782992, 0.5413880912001547, 0.731840942842999, 0.8396640419159087] coeffs [0.50847073 1.15392272 1.2059733  1.1340391  1.03196719]
        (
            "Dec30_15_34_resnet101_fpncatv2_256_512_fold0_fp16_pseudo_crops.pth",
            [0.50847073, 1.15392272, 1.2059733, 1.1340391, 1.03196719],
        ),
    ]

    fold_1_models_dict = [
        # (
        #     "Dec16_18_59_densenet201_fpncatv2_256_512_fold1_fp16_crops.pth",
        #     [0.64202075, 1.04641224, 1.23015655, 1.03203408, 1.12505602],
        # ),
        # (
        #     "Dec17_01_52_resnet34_unet_v2_512_fold1_fp16_crops.pth",
        #     [0.69605759, 0.89963168, 0.9232137, 0.92938775, 0.94460875],
        # ),
        (
            "Dec22_22_24_seresnext50_unet_v2_512_fold1_fp16_crops.pth",
            [0.54324459, 1.76890163, 1.20782899, 0.85128004, 0.83100698],
        ),
        (
            "Dec31_02_09_resnet34_unet_v2_512_fold1_fp16_pseudo_crops.pth",
            # Maybe suboptimal
            [0.48269921, 1.22874469, 1.38328066, 0.96695393, 0.91348539],
        ),
        (
            "Dec31_03_55_densenet201_fpncatv2_256_512_fold1_fp16_pseudo_crops.pth",
            [0.48804137, 1.14809462, 1.24851827, 1.11798428, 1.00790482]
        )
    ]

    fold_2_models_dict = [
        # (
        #     "Dec17_19_19_resnet34_unet_v2_512_fold2_fp16_crops.pth",
        #     [0.65977938, 1.50252452, 0.97098732, 0.74048182, 1.08712367],
        # ),
        # 0.7674290884579319 0.8107652756500724 0.7488564368041575 [0.9228529822124596, 0.5900700454049471, 0.736806959757804, 0.8292099253270483] coeffs [0.34641084 1.63486251 1.14186036 0.86668715 1.12193125]
        (
            "Dec17_19_12_inceptionv4_fpncatv2_256_512_fold2_fp16_crops.pth",
            [0.34641084, 1.63486251, 1.14186036, 0.86668715, 1.12193125],
        ),
        # 0.7683650436367244 0.8543981047493 0.7314937317313349 [0.9248137307721042, 0.5642011151253543, 0.7081016179096937, 0.831720163492164] coeffs [0.51277498 1.4475809  0.8296623  0.97868596 1.34180805]
        (
            "Dec27_14_08_densenet169_unet_v2_512_fold2_fp16_crops.pth",
            [0.55429115, 1.34944309, 1.1087044, 0.89542089, 1.17257541],
        ),
        (
            "Dec31_12_45_resnet34_unet_v2_512_fold2_fp16_pseudo_crops.pth",
            # Copied from Dec17_19_19_resnet34_unet_v2_512_fold2_fp16_crops
            [0.65977938, 1.50252452, 0.97098732, 0.74048182, 1.08712367],
        )
    ]

    fold_3_models_dict = [
        (
            "Dec15_23_24_resnet34_unet_v2_512_fold3_crops.pth",
            [0.84090623, 1.02953555, 1.2526516, 0.9298182, 0.94053529],
        ),
        # (
        #     "Dec18_12_49_resnet34_unet_v2_512_fold3_fp16_crops.pth",
        #     [0.55555375, 1.18287119, 1.10997173, 0.85927596, 1.18145368],
        # ),
        # (
        #     "Dec19_14_59_efficientb4_fpncatv2_256_512_fold3_fp16_crops.pth",
        #     [0.59338243, 1.17347438, 1.186104, 1.06860638, 1.03041829],
        # ),
        (
            "Dec21_11_50_seresnext50_unet_v2_512_fold3_fp16_crops.pth",
            [0.43108046, 1.30222898, 1.09660616, 0.94958969, 1.07063753],
        ),
        (
            "Dec31_18_17_efficientb4_fpncatv2_256_512_fold3_fp16_pseudo_crops.pth",
            # Copied from Dec19_14_59_efficientb4_fpncatv2_256_512_fold3_fp16_crops
            [0.59338243, 1.17347438, 1.186104, 1.06860638, 1.03041829]
        )
    ]

    fold_4_models_dict = [
        (
            "Dec19_06_18_resnet34_unet_v2_512_fold4_fp16_crops.pth",
            [0.83915734, 1.02560309, 0.77639015, 1.17487775, 1.05632771],
        ),
        (
            "Dec27_14_37_resnet101_unet_v2_512_fold4_fp16_crops.pth",
            [0.57414314, 1.19599486, 1.05561912, 0.98815567, 1.2274592],
        ),
    ]

    infos = []

    resize = A.Resize(1024, 1024)
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
    transform = A.Compose([resize, normalize])

    # Very dumb way but it matches 1:1 with inferencing
    pre, post = read_image(pre_image), read_image(post_image)
    image = np.dstack([pre, post])
    image = transform(image=image)["image"]
    pre_image = image[..., 0:3]
    post_image = image[..., 3:6]

    models = []
    for models_dict in [
        fold_0_models_dict,
        fold_1_models_dict,
        fold_2_models_dict,
        fold_3_models_dict,
        fold_4_models_dict,
    ]:
        for checkpoint, weights in models_dict:
            model, info = weighted_model(checkpoint, weights, activation="model")
            models.append(model)
            infos.append(info)

    model = Ensembler(models, outputs=[OUTPUT_MASK_KEY])
    model = HFlipTTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
    model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-128, +128], average=True)
    model = model.eval()

    df = pd.DataFrame.from_records(infos)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", -1)

    print(df)
    print("score        ", df["score"].mean(), df["score"].std())
    print("localization ", df["localization"].mean(), df["localization"].std())
    print("damage       ", df["damage"].mean(), df["damage"].std())

    input_image = tensor_from_rgb_image(np.dstack([pre_image, post_image])).unsqueeze(0)

    if use_gpu:
        print("Using GPU for inference")
        input_image = input_image.cuda()
        model = model.cuda()

    output = model(input_image)
    masks = output[OUTPUT_MASK_KEY]
    predictions = to_numpy(masks.squeeze(0)).astype(np.float32)

    if save_raw:
        np.save(raw_predictions_file, predictions)

    localization_image, damage_image = make_predictions_naive(predictions)

    if color_mask:
        localization_image = colorize_mask(localization_image)
        localization_image.save(localization_fname)

        damage_image = colorize_mask(damage_image)
        damage_image.save(damage_fname)
    else:
        cv2.imwrite(localization_fname, localization_image)
        cv2.imwrite(damage_fname, damage_image)

    print("Saved output to ", localization_fname, damage_fname)

    done = time.time()
    elapsed = done - start
    print("Inference time", elapsed, "(s)")


if __name__ == "__main__":
    main()