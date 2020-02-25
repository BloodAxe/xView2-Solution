#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          #
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, #
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################


import json
from typing import List, Tuple

from shapely import wkt
from shapely.geometry import Polygon
import numpy as np
from cv2 import fillPoly, imwrite

from PIL import Image


def open_json(json_file_path):
    """
    :param json_file_path: path to open inference json file
    :returns: the json data dictionary of localized polygon and their classifications 
    """

    with open(json_file_path) as jf:
        json_data = json.load(jf)
        inference_data = json_data["features"]["xy"]
        return inference_data


def resize_mask_one_hot(mask, size):
    import albumentations as A
    import cv2

    mask = np.squeeze(mask,-1)
    one_hot_mask = (np.arange(mask.max() + 1) == mask[..., None]).astype(np.float32)

    resize_op = A.Resize(size[0], size[1], interpolation=cv2.INTER_LINEAR)
    image_resized = resize_op(image=one_hot_mask)["image"]

    mask = np.argmax(image_resized, axis=2).astype(mask.dtype)
    return mask


def create_image(inference_data) -> np.ndarray:
    """
    :params inference_data: json data dictionary of localized polygon and their classifications
    :returns: an numpy array of 8-bit grey scale image with polygons filled in according to the key provided
    """

    damage_key = {"un-classified": 1, "no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}

    mask_img = np.zeros((1024, 1024, 1), np.uint8)

    for poly in inference_data:
        damage = poly["properties"].get("subtype", "no-damage")
        coords = wkt.loads(poly["wkt"])
        poly_np = np.array(coords.exterior.coords, np.int32)

        fillPoly(mask_img, [poly_np], damage_key[damage])

    return mask_img


def create_instance_image(inference_data) -> Tuple[np.ndarray, List[int]]:
    """
    :params inference_data: json data dictionary of localized polygon and their classifications
    :returns: an numpy array of 16-bit grey scale image with polygons filled in according to the key provided
    """

    damage_key = {"un-classified": 1, "no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}

    mask_img = np.zeros((1024, 1024), np.uint16)

    damage_labels = []
    for poly_index, poly in enumerate(inference_data):
        damage = poly["properties"].get("subtype", "no-damage")
        damage_label = damage_key[damage]
        damage_labels.append(damage_label)

        coords = wkt.loads(poly["wkt"])
        poly_np = np.array(coords.exterior.coords, np.int32)

        fillPoly(mask_img, [poly_np], poly_index + 1)

    return mask_img, damage_labels


def colorize_mask(mask, color_map=None):
    """
    Attaches a color palette to a PIL image. So long as the image is saved as a PNG, it will render visibly using the
    provided color map.
    :param mask: PIL image whose values are only 0 to 4 inclusive
    :param color_map: np.ndarray or list of 3-tuples with 5 rows
    :return:
    """
    if len(mask.shape) == 3:
        mask = np.squeeze(mask, -1)
    mask = Image.fromarray(mask, "L")

    color_map = color_map or np.array(
        [
            (0, 0, 0),  # 0=background
            (0, 255, 0),  # no damage (or just 'building' for localization) (green)
            (255, 255, 0),  # minor damage (yellow)
            (255, 128, 0),  # major damage (red)
            (255, 0, 0),  # destroyed (red)
            (127, 127, 127),  # Unlabeled
        ]
    )
    assert color_map.shape == (6, 3)
    mask.putpalette(color_map.astype(np.uint8))
    return mask


def make_rgb_image(mask):

    color_map = np.array(
        [
            (0, 0, 0),  # 0=background
            (0, 255, 0),  # no damage (or just 'building' for localization) (green)
            (255, 255, 0),  # minor damage (yellow)
            (255, 128, 0),  # major damage (red)
            (255, 0, 0),  # destroyed (red)
            (127, 127, 127),  # Unlabeled
        ], dtype=np.uint8
    )

    mask_rgb = color_map[mask]
    return mask_rgb

def save_image(polygons, output_path):
    """
    :param polygons: np array with filled in polygons from create_image()
    :param output_path: path to save the final output inference image
    """
    mask = colorize_mask(polygons)
    mask.save(output_path)

    # Output the filled in polygons to an image file
    # imwrite(output_path, polygons)


def create_inference_image(json_input_path, image_output_path):
    """
    :param json_input_path: Path to output inference json file
    :param image_outut_pat: Path to save the final inference image
    """

    # Getting the inference data from the localization and classification
    inference_data = open_json(json_input_path)

    # Filling in the polygons and readying the image format
    polygon_array = create_image(inference_data)

    # Saving the image to the desired location
    save_image(polygon_array, image_output_path)
    return polygon_array


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""inference_image_output.py: Takes the inference localization and classification final outputs in json from and outputs an image ready to be scored based off the challenge parameters"""
    )
    parser.add_argument(
        "--input", required=True, metavar="/path/to/final/inference.json", help="Full path to the final inference json"
    )
    parser.add_argument(
        "--output", required=True, metavar="/path/to/inference.png", help="Full path to save the image to"
    )

    args = parser.parse_args()

    # Creating the scoring image
    create_inference_image(args.input, args.output)
