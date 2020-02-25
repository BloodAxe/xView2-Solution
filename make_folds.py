import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default="c:\\datasets\\xview2")
    args = parser.parse_args()

    data_dir = args.data_dir

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df = df.sort_values(by="sample_id")
    df["fold"] = -1

    df_pre = df[df["event_type"] == "pre"].copy()
    df_post = df[df["event_type"] == "post"].copy()

    # Use only post samples to split data

    # destroyed_buildings,destroyed_pixels,event_name,event_type,light_damaged_buildings,light_damaged_pixels,medium_damaged_buildings,medium_damaged_pixels,non_damaged_buildings,non_damaged_pixels,sample_id
    y = np.column_stack(
        [
            df_post["non_damaged_buildings"].values > 0,
            df_post["light_damaged_buildings"].values > 0,
            df_post["medium_damaged_buildings"].values > 0,
            df_post["destroyed_buildings"].values > 0,
            LabelEncoder().fit_transform(df_post["event_name"].tolist()),
        ]
    )

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    folds = np.ones(len(y), dtype=int) * -1

    for fold, (train_index, test_index) in enumerate(mskf.split(df_post, y)):
        folds[test_index] = fold

    df_pre["fold"] = folds
    df_post["fold"] = folds
    df = pd.concat((df_pre, df_post))

    df.to_csv(os.path.join(data_dir, "train_folds.csv"), index=None)


if __name__ == "__main__":
    main()
