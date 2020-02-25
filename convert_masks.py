import argparse
import os

import pandas as pd
from pytorch_toolbelt.utils import fs
from skimage.measure import label
from tqdm import tqdm

from xview.utils.inference_image_output import create_inference_image


def convert_dir(dir, folder):
    jsons_dir = os.path.join(dir, "labels")
    masks_dir = os.path.join(dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    jsons = [fname for fname in fs.find_in_dir(jsons_dir) if fname.endswith(".json")]

    items = []
    for json_fname in tqdm(jsons):
        mask_fname = os.path.join(masks_dir, fs.id_from_fname(json_fname) + ".png")
        mask = create_inference_image(json_fname, mask_fname)

        non_damaged_mask = mask == 1
        light = mask == 2
        medium = mask == 3
        destroyed = mask == 4

        non_damaged_pixels = non_damaged_mask.sum()
        light_pixels = light.sum()
        medium_pixels = medium.sum()
        destroyed_pixels = destroyed.sum()

        # guatemala-volcano_00000000_post_disaster
        event_name, sample_id, event_type, disaster = fs.id_from_fname(json_fname).split("_")
        assert disaster == "disaster"

        image_id = fs.id_from_fname(json_fname)
        items.append(
            {
                "image_fname": os.path.join(folder, "images", image_id + ".png"),
                "mask_fname": os.path.join(folder, "masks", image_id + ".png"),
                "folder": folder,
                "image_id": image_id,
                "event_name": event_name,
                "sample_id": sample_id,
                "event_type": event_type,
                "non_damaged_pixels": non_damaged_pixels,
                "light_damaged_pixels": light_pixels,
                "medium_damaged_pixels": medium_pixels,
                "destroyed_pixels": destroyed_pixels,
                "non_damaged_buildings": label(non_damaged_mask, return_num=True)[1],
                "light_damaged_buildings": label(light, return_num=True)[1],
                "medium_damaged_buildings": label(medium, return_num=True)[1],
                "destroyed_buildings": label(destroyed, return_num=True)[1],
            }
        )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default="c:\\datasets\\xview2")
    args = parser.parse_args()

    data_dir = args.data_dir

    train_dir = os.path.join(data_dir, "train")
    tier3_dir = os.path.join(data_dir, "tier3")

    items = []

    items += convert_dir(train_dir, folder="train")
    items += convert_dir(tier3_dir, folder="tier3")

    df = pd.DataFrame.from_records(items)
    df.to_csv(os.path.join(data_dir, "train.csv"), index=None)


if __name__ == "__main__":
    main()
