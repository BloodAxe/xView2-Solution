import argparse
import os
from collections import defaultdict
import numpy as np

from xview.rounder import OptimizedRounder

from pytorch_toolbelt.utils import fs
import pandas as pd
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("-w", "--workers", type=int, default=1, help="")
    parser.add_argument("-dd", "--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("-a", "--activation", type=str, default="pre", help="")
    args = parser.parse_args()

    targets = fs.find_in_dir(os.path.join(args.data_dir, "tier3", "masks")) + fs.find_in_dir(
        os.path.join(args.data_dir, "train", "masks")
    )
    targets_post = dict((fs.id_from_fname(fname), fname) for fname in targets if "_post_" in fname)

    df = defaultdict(list)

    current_time = datetime.now().strftime("%b%d_%H_%M")

    print("Checkpoints ", args.checkpoints)
    print("Activation  ", args.activation)

    for model_checkpoint in args.checkpoints:
        model_checkpoint = fs.auto_file(model_checkpoint)
        predictions_dir = os.path.join(
            os.path.dirname(model_checkpoint), fs.id_from_fname(model_checkpoint) + "_oof_predictions"
        )

        prediction_files = fs.find_in_dir(predictions_dir)
        prediction_files_post = dict(
            (fs.id_from_fname(fname), fname) for fname in prediction_files if "_post_" in fname
        )

        y_true_filenames = [targets_post[image_id_post] for image_id_post in prediction_files_post.keys()]
        y_pred_filenames = [prediction_files_post[image_id_post] for image_id_post in prediction_files_post.keys()]

        rounder = OptimizedRounder(workers=args.workers, apply_softmax=args.activation)

        raw_score, raw_localization_f1, raw_damage_f1, raw_damage_f1s = rounder.predict(
            y_pred_filenames, y_true_filenames, np.array([1, 1, 1, 1, 1], dtype=np.float32)
        )

        rounder.fit(y_pred_filenames, y_true_filenames)

        score, localization_f1, damage_f1, damage_f1s = rounder.predict(
            y_pred_filenames, y_true_filenames, rounder.coefficients()
        )

        print(rounder.coefficients())

        df["checkpoint"].append(fs.id_from_fname(model_checkpoint))
        df["coefficients"].append(rounder.coefficients())
        df["samples"].append(len(y_true_filenames))

        df["raw_score"].append(raw_score)
        df["raw_localization"].append(raw_localization_f1)
        df["raw_damage"].append(raw_damage_f1)

        df["opt_score"].append(score)
        df["opt_localization"].append(localization_f1)
        df["opt_damage"].append(damage_f1)

        dataframe = pd.DataFrame.from_dict(df)
        dataframe.to_csv(f"optimized_weights_{current_time}.csv", index=None)
        print(df)


if __name__ == "__main__":
    main()
