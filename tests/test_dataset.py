import cv2

from xview.augmentations import old_post_transform_augs, light_post_image_transform, medium_post_transform_augs
from xview.dataset import make_dual_dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_dataset():
    df = pd.read_csv("../train_folds.csv")
    train_df = make_dual_dataframe(df)

    non_damaged_buildings = train_df["non_damaged_buildings_post"].values
    light_damaged_buildings = train_df["light_damaged_buildings_post"].values
    medium_damaged_buildings = train_df["medium_damaged_buildings_post"].values
    destroyed_buildings = train_df["destroyed_buildings_post"].values

    non_damaged_pixels = train_df["non_damaged_pixels_post"].values
    light_damaged_pixels = train_df["light_damaged_pixels_post"].values
    medium_damaged_pixels = train_df["medium_damaged_pixels_post"].values
    destroyed_pixels = train_df["destroyed_pixels_post"].values

    print(
        non_damaged_buildings.sum(),
        light_damaged_buildings.sum(),
        medium_damaged_buildings.sum(),
        destroyed_buildings.sum(),
    )
    print(
        (1024 * 1024) * len(train_df)
        - non_damaged_pixels.sum()
        - light_damaged_pixels.sum()
        - medium_damaged_pixels.sum()
        - destroyed_pixels.sum(),
        non_damaged_pixels.sum(),
        light_damaged_pixels.sum(),
        medium_damaged_pixels.sum(),
        destroyed_pixels.sum(),
    )


def test_post_transform():
    image = cv2.imread("guatemala-volcano_00000000_post_disaster.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    post_transform = old_post_transform_augs()

    image_acc = image.astype(np.long)

    n = 1000

    for i in range(n):
        image_t = post_transform(image=image)["image"]
        image_acc += image_t

    image_acc = (image_acc * (1. / n)).astype(np.uint8)

    plt.figure()
    plt.imshow(image)
    plt.show()

    plt.figure()
    plt.imshow(image_acc)
    plt.show()



def test_light_post_image_transform():
    image = cv2.imread("guatemala-volcano_00000000_post_disaster.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    post_transform = light_post_image_transform()

    image_acc = image.astype(np.long)

    n = 1000

    for i in range(n):
        image_t = post_transform(image=image)["image"]
        image_acc += image_t

    image_acc = (image_acc * (1. / n)).astype(np.uint8)

    plt.figure()
    plt.imshow(image)
    plt.show()

    plt.figure()
    plt.imshow(image_acc)
    plt.show()



def test_medium_post_transform_augs():
    image = cv2.imread("guatemala-volcano_00000000_post_disaster.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    plt.figure()
    plt.imshow(image)
    plt.show()

    post_transform = medium_post_transform_augs()

    k = 10
    for i in range(k):
        image_t = post_transform(image=image)["image"]

        plt.figure()
        plt.imshow(image_t)
        plt.show()

    image_acc = image.astype(np.long)

    n = 100

    for i in range(n):
        image_t = post_transform(image=image)["image"]
        image_acc += image_t

    image_acc = (image_acc * (1. / n)).astype(np.uint8)


    plt.figure()
    plt.imshow(image_acc)
    plt.show()
