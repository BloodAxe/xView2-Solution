import argparse
import os

import cv2
from skimage.measure import label
from tqdm import tqdm
import pandas as pd

from pytorch_toolbelt.utils import fs
import numpy as np

from xview.dataset import make_dual_dataframe, read_image
from xview.utils.inference_image_output import create_inference_image, open_json, create_instance_image
from PIL import Image


def bbox1(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]) + 1, np.min(a[1]), np.max(a[1]) + 1
    return bbox


def convert_dir(df: pd.DataFrame, dir) -> pd.DataFrame:
    crops_dir = os.path.join(dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    building_crops = []

    global_crop_index = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_fname_pre = read_image(os.path.join(dir, row["image_fname_pre"]))
        image_fname_post = read_image(os.path.join(dir, row["image_fname_post"]))

        mask_fname_post = row["mask_fname_post"]
        json_fname_post = fs.change_extension(mask_fname_post.replace("masks", "labels"), ".json")
        inference_data = open_json(os.path.join(dir, json_fname_post))
        instance_image, labels = create_instance_image(inference_data)

        for label_index, damage_label in zip(range(1, instance_image.max() + 1), labels):
            try:
                instance_mask = instance_image == label_index
                rmin, rmax, cmin, cmax = bbox1(instance_mask)

                max_size = max(rmax - rmin, cmax - cmin)
                if max_size < 16:
                    print(
                        "Skipping crop since it's too small",
                        fs.id_from_fname(mask_fname_post),
                        "label_index",
                        label_index,
                        "min_size",
                        max_size
                    )
                    continue

                rpadding = (rmax - rmin) // 4
                cpadding = (cmax - cmin) // 4

                pre_crop = image_fname_pre[
                    max(0, rmin - rpadding) : rmax + rpadding, max(0, cmin - cpadding) : cmax + cpadding
                ]
                post_crop = image_fname_post[
                    max(0, rmin - rpadding) : rmax + rpadding, max(0, cmin - cpadding) : cmax + cpadding
                ]

                image_id_pre = row["image_id_pre"]
                image_id_post = row["image_id_post"]

                pre_crop_fname = f"{global_crop_index:06}_{image_id_pre}.png"
                post_crop_fname = f"{global_crop_index:06}_{image_id_post}.png"
                global_crop_index += 1

                cv2.imwrite(os.path.join(crops_dir, pre_crop_fname), pre_crop)
                cv2.imwrite(os.path.join(crops_dir, post_crop_fname), post_crop)

                building_crops.append(
                    {
                        "pre_crop_fname": pre_crop_fname,
                        "post_crop": post_crop_fname,
                        "label": damage_label,
                        "event_name": row["event_name_post"],
                        "fold": row["fold_post"],
                        "rmin": rmin,
                        "rmax": rmax,
                        "cmin": cmin,
                        "cmax": cmax,
                        "max_size": max_size,
                        "rpadding": rpadding,
                        "cpadding": cpadding
                    }
                )
            except Exception as e:
                print(e)
                print(mask_fname_post)

    df = pd.DataFrame.from_records(building_crops)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default="c:\\datasets\\xview2")
    args = parser.parse_args()

    data_dir = args.data_dir

    df = pd.read_csv(os.path.join(data_dir, "train_folds.csv"))
    df = make_dual_dataframe(df)

    df_crops = convert_dir(df, data_dir)
    df_crops.to_csv(os.path.join(data_dir, "train_crops.csv"), index=None)


if __name__ == "__main__":
    main()
