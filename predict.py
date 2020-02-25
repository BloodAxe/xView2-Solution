import argparse
import os
from collections import defaultdict
from datetime import datetime

import torch
import pandas as pd
from pytorch_toolbelt.utils import fs

from xview.dataset import get_test_dataset, OUTPUT_MASK_KEY
from xview.inference import Ensembler, model_from_checkpoint, run_inference_on_dataset, ApplySoftmaxTo, MultiscaleTTA, \
    HFlipTTA, D4TTA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tta", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size during training, e.g. -b 64")
    parser.add_argument("-w", "--workers", type=int, default=0, help="")
    parser.add_argument("-dd", "--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("-p", "--postprocessing", type=str, default="dominant")
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--activation", default="model", type=str)
    parser.add_argument("--weights", default=None, type=float, nargs="+")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--align", action="store_true")

    args = parser.parse_args()

    workers = args.workers
    data_dir = args.data_dir
    fast = args.fast
    tta = args.tta
    image_size = args.size, args.size
    model_checkpoints = args.models
    batch_size = args.batch_size
    activation_after = args.activation
    fp16 = args.fp16
    align = args.align
    postprocessing=args.postprocessing
    weights = args.weights
    assert weights is None or len(weights) == 5

    current_time = datetime.now().strftime("%b%d_%H_%M")
    if args.output_dir is None and len(model_checkpoints) == 1:
        output_dir = os.path.join(
            os.path.dirname(model_checkpoints[0]), fs.id_from_fname(model_checkpoints[0]) + "_test_predictions"
        )
        if weights is not None:
            output_dir += "_weighted"
        if tta is not None:
            output_dir += f"_{tta}"
    else:
        output_dir = args.output_dir or f"output_dir_{current_time}"

    print("Size",       image_size)
    print("Output dir", output_dir)
    print("Postproc  ", postprocessing)
    # Load models

    models = []
    infos = []
    for model_checkpoint in model_checkpoints:
        try:
            model, info = model_from_checkpoint(
                fs.auto_file(model_checkpoint), tta=None, activation_after=activation_after, report=False
            )
            models.append(model)
            infos.append(info)
        except Exception as e:
            print(e)
            print(model_checkpoint)
            return

    df = pd.DataFrame.from_records(infos)
    print(df)

    print("score        ", df["score"].mean(), df["score"].std())
    print("localization ", df["localization"].mean(), df["localization"].std())
    print("damage       ", df["damage"].mean(), df["damage"].std())

    if len(models) > 1:
        model = Ensembler(models, [OUTPUT_MASK_KEY])
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

        if tta == "multiscale_d4":
            print(f"Using {tta}")
            model = D4TTA(model, outputs=[OUTPUT_MASK_KEY], average=True)
            model = MultiscaleTTA(model, outputs=[OUTPUT_MASK_KEY], size_offsets=[-128, +128], average=True)

        if activation_after == "tta":
            model = ApplySoftmaxTo(model, OUTPUT_MASK_KEY)

    else:
        model = models[0]

    test_ds = get_test_dataset(data_dir=data_dir, image_size=image_size, fast=fast, align_post=align)

    run_inference_on_dataset(
        model=model,
        dataset=test_ds,
        output_dir=output_dir,
        batch_size=batch_size,
        workers=workers,
        weights=weights,
        fp16=fp16,
        postprocessing=postprocessing,
    )


if __name__ == "__main__":
    main()
