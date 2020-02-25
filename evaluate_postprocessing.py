import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool

import cv2
from tqdm import tqdm

from xview.dataset import read_mask
from xview.metric import CompetitionMetricCallback
from xview.postprocessing import make_predictions_dominant, make_predictions_naive, make_predictions_floodfill

from pytorch_toolbelt.utils import fs
import pandas as pd
from datetime import datetime
import numpy as np


def _compute_fn(args, postprocessing_fn):
    xi, yi = args
    dmg_pred = np.load(xi)
    dmg_true = read_mask(yi)

    loc_pred, dmg_pred = postprocessing_fn(dmg_pred)

    if loc_pred.shape[0] != 1024:
        loc_pred = cv2.resize(loc_pred, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
        dmg_pred = cv2.resize(dmg_pred, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

    row = CompetitionMetricCallback.get_row_pair(loc_pred, dmg_pred, dmg_true, dmg_true)
    return row


def optimize_postprocessing(y_pred_filenames, y_true_filenames, workers: int, postprocessing_fn):
    input = list(zip(y_pred_filenames, y_true_filenames))

    all_rows = []
    process = partial(_compute_fn, postprocessing_fn=postprocessing_fn)
    with Pool(workers) as wp:
        for row in tqdm(wp.imap_unordered(process, input, chunksize=8), total=len(input)):
            all_rows.append(row)

    return CompetitionMetricCallback.compute_metrics(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("-w", "--workers", type=int, default=0, help="")
    parser.add_argument("-dd", "--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    targets = fs.find_in_dir(os.path.join(args.data_dir, "tier3", "masks")) + fs.find_in_dir(
        os.path.join(args.data_dir, "train", "masks")
    )
    targets_post = dict((fs.id_from_fname(fname), fname) for fname in targets if "_post_" in fname)

    df = defaultdict(list)

    postprocessings = {
        "naive": make_predictions_naive,
        "dominant": make_predictions_dominant,
        "floodfill": make_predictions_floodfill,
    }

    for predictions_dir in args.predictions:
        try:
            prediction_files = fs.find_in_dir(predictions_dir)
            prediction_files_post = dict(
                (fs.id_from_fname(fname), fname) for fname in prediction_files if "_post_" in fname
            )

            y_true_filenames = [targets_post[image_id_post] for image_id_post in prediction_files_post.keys()]
            y_pred_filenames = [prediction_files_post[image_id_post] for image_id_post in prediction_files_post.keys()]

            for name, fn in postprocessings.items():
                score, localization_f1, damage_f1, damage_f1s = optimize_postprocessing(
                    y_pred_filenames, y_true_filenames, postprocessing_fn=fn, workers=args.workers
                )

                print(name, score)

                df["samples"].append(len(y_pred_filenames))
                df["predictions_dir"].append(predictions_dir)
                df["postprocessing"].append(name)
                df["score"].append(score)
                df["localization_f1"].append(localization_f1)
                df["damage_f1"].append(damage_f1)
        except Exception as e:
            print("Failed to process", predictions_dir, e)

    df = pd.DataFrame.from_dict(df)
    print(df)

    current_time = datetime.now().strftime("%b%d_%H_%M")

    df.to_csv(f"postprocessing_eval_{current_time}.csv", index=None)


if __name__ == "__main__":
    main()
