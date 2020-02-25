import argparse
import os

import torch
from pytorch_toolbelt.utils import fs

from xview.dataset import get_datasets
from xview.inference import model_from_checkpoint, run_inference_on_dataset_oof
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, nargs="+")
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tta", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size during training, e.g. -b 64")
    parser.add_argument("-w", "--workers", type=int, default=0, help="")
    parser.add_argument("-dd", "--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--fold", default=None, type=int)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--activation", default="model", type=str)
    parser.add_argument("--align", action="store_true")

    args = parser.parse_args()

    fp16 = args.fp16
    activation = args.activation

    average_score = []
    average_dmg = []
    average_loc = []

    for model_checkpoint in args.model:
        model_checkpoint = fs.auto_file(model_checkpoint)
        checkpoint = torch.load(model_checkpoint)

        print("Model        :", model_checkpoint)
        print(
            "Metrics      :",
            checkpoint["epoch_metrics"]["valid"]["weighted_f1"],
            checkpoint["epoch_metrics"]["valid"]["weighted_f1/localization_f1"],
            checkpoint["epoch_metrics"]["valid"]["weighted_f1/damage_f1"],
        )

        workers = args.workers
        data_dir = args.data_dir
        fast = args.fast
        tta = args.tta
        no_save = args.no_save
        image_size = args.size or checkpoint["checkpoint_data"]["cmd_args"]["size"]
        batch_size = args.batch_size or checkpoint["checkpoint_data"]["cmd_args"]["batch_size"]
        fold = args.fold or checkpoint["checkpoint_data"]["cmd_args"]["fold"]
        align = args.align

        print("Image size :", image_size)
        print("Fold       :", fold)
        print("Align      :", align)
        print("Workers    :", workers)
        print("Save       :", not no_save)

        output_dir = None
        if not no_save:
            output_dir = args.output_dir or os.path.join(
                os.path.dirname(model_checkpoint), fs.id_from_fname(model_checkpoint) + "_oof_predictions"
            )
            print("Output dir :", output_dir)

        # Load models
        model, info = model_from_checkpoint(model_checkpoint, tta=tta, activation_after=None, report=False)
        print(info)
        _, valid_ds, _ = get_datasets(data_dir=data_dir, image_size=(image_size, image_size), fast=fast, fold=fold, align_post=align)

        score, localization_f1, damage_f1, damage_f1s = run_inference_on_dataset_oof(
            model=model,
            dataset=valid_ds,
            output_dir=output_dir,
            batch_size=batch_size,
            workers=workers,
            save=not no_save,
            fp16=fp16
        )

        average_score.append(score)
        average_dmg.append(damage_f1)
        average_loc.append(localization_f1)

        print("Score        :", score)
        print("Localization :", localization_f1)
        print("Damage       :", damage_f1)
        print("Per class    :", damage_f1s)
        print()

    print("Average")
    if len(average_score) > 1:
        print("Score        :", np.mean(average_score), np.std(average_score))
        print("Localization :", np.mean(average_loc), np.std(average_loc))
        print("Damage       :", np.mean(average_dmg), np.std(average_dmg))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
