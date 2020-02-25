import argparse

import pandas as pd
import torch
from pytorch_toolbelt.utils import fs

from xview.dataset import OUTPUT_MASK_KEY, get_test_dataset
from xview.inference import (
    model_from_checkpoint,
    ApplyWeights,
    Ensembler,
    ApplySoftmaxTo,
    MultiscaleTTA,
    HFlipTTA,
    D4TTA,
    run_inference_on_dataset,
)


def weighted_model(checkpoint_fname: str, weights, activation: str):
    model, info = model_from_checkpoint(fs.auto_file(checkpoint_fname, where="models"), activation_after=activation, report=False)
    model = ApplyWeights(model, weights)
    return model, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, default="models/predict_37_weighted")
    parser.add_argument("--tta", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size during training, e.g. -b 64")
    parser.add_argument("-w", "--workers", type=int, default=0, help="")
    parser.add_argument("-dd", "--data-dir", type=str, default="c:\\datasets\\xview2", help="Data directory")
    parser.add_argument("-p", "--postprocessing", type=str, default=None)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--activation", default="model", type=str)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    workers = args.workers
    data_dir = args.data_dir
    tta = args.tta
    image_size = args.size, args.size
    batch_size = args.batch_size
    activation_after = args.activation
    fp16 = args.fp16
    postprocessing = args.postprocessing
    output_dir = args.output_dir

    print("Size      ", image_size)
    print("Output dir", output_dir)
    print("Postproc  ", postprocessing)

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
    models = []

    for models_dict in [
        fold_0_models_dict,
        fold_1_models_dict,
        fold_2_models_dict,
        fold_3_models_dict,
        fold_4_models_dict,
    ]:
        for checkpoint, weights in models_dict:
            model, info = weighted_model(checkpoint, weights, activation_after)
            models.append(model)
            infos.append(info)

    model = Ensembler(models, outputs=[OUTPUT_MASK_KEY])

    df = pd.DataFrame.from_records(infos)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", -1)

    print(df)
    print("score        ", df["score"].mean(), df["score"].std())
    print("localization ", df["localization"].mean(), df["localization"].std())
    print("damage       ", df["damage"].mean(), df["damage"].std())

    if activation_after == "ensemble":
        model = ApplySoftmaxTo(model, OUTPUT_MASK_KEY)
        print("Applying activation after ensemble")

    if tta == "multiscale":
        print(f"Using {tta}")
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-128, +128], average=True)

    if tta == "flip":
        print(f"Using {tta}")
        model = HFlipTTA(model, outputs=[OUTPUT_MASK_KEY], average=True)

    if tta == "flipscale":
        print(f"Using {tta}")
        model = HFlipTTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-128, +128], average=True)

    if tta == "flipscale2":
        print(f"Using {tta}")
        model = HFlipTTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-256, -128, +128, +256], average=True)

    if tta == "multiscale_d4":
        print(f"Using {tta}")
        model = D4TTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
        model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-128, +128], average=True)

    if tta is not None:
        output_dir += "_" + tta

    if activation_after == "tta":
        model = ApplySoftmaxTo(model, OUTPUT_MASK_KEY)
        print("Applying activation after TTA")

    test_ds = get_test_dataset(data_dir=data_dir, image_size=image_size)

    run_inference_on_dataset(
        model=model,
        dataset=test_ds,
        output_dir=output_dir,
        batch_size=batch_size,
        workers=workers,
        fp16=fp16,
        postprocessing=postprocessing,
        save_pseudolabels=False,
        cpu=False
    )


if __name__ == "__main__":
    main()
