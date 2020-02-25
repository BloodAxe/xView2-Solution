import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

__all__ = ["make_predictions_dominant", "make_predictions_naive", "make_predictions_floodfill"]

from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential

from xview.dataset import UNLABELED_SAMPLE


def make_pseudolabeling_target(damage_probs:np.ndarray, ratio_threshold=1.5):
    damage_probs = damage_probs.copy()
    class_index = np.argmax(damage_probs, axis=0)

    sorted_probs = np.sort(-damage_probs, axis=0)

    ratio = sorted_probs[0] / sorted_probs[1]

    confident_classes = ratio > ratio_threshold
    class_index[~confident_classes] = UNLABELED_SAMPLE
    return class_index


def make_predictions_naive(damage_probs: np.ndarray):
    loc_pred = np.stack((damage_probs[0, ...], np.sum(damage_probs[1:, ...], axis=0)))
    loc_cls = np.argmax(loc_pred, axis=0)

    # After we have 'fixed' localization predictions, we must zero-out probabilities for damage probs
    damage_probs = damage_probs.copy()
    damage_probs[0, loc_cls > 0] = 0
    dmg_cls = np.argmax(damage_probs, axis=0)

    dmg_cls[dmg_cls == 0] = 1  # Fill remaining with damage type 1 (no damage)
    return loc_cls.astype(np.uint8), dmg_cls.astype(np.uint8)



def make_predictions_dominant(
    damage_probs: np.ndarray, min_size=32, assign_dominant=True, max_building_area=2048, min_solidity=0.75
):
    loc_pred = np.stack((damage_probs[0, ...], np.sum(damage_probs[1:, ...], axis=0)))
    loc_cls = np.argmax(loc_pred, axis=0)

    # After we have 'fixed' localization predictions, we must zero-out probabilities for damage probs
    damage_probs = damage_probs.copy()
    damage_probs[0, loc_cls > 0] = 0
    dmg_cls = np.argmax(damage_probs, axis=0)

    buildings = label(loc_cls)

    if min_size is not None:
        # If there are any objects at all
        if buildings.max() > 0:
            buildings = remove_small_objects(buildings, min_size=min_size)
            buildings, _, _ = relabel_sequential(buildings)
            loc_cls = buildings > 0
            dmg_cls[~loc_cls] = 0

    if assign_dominant:
        building_props = regionprops(buildings)
        classes = list(range(1, 5))
        for index, region in enumerate(building_props):
            region_label, area, solidity = region["label"], region["area"], region["solidity"]

            region_mask = buildings == region_label

            if area < max_building_area or solidity > min_solidity:
                label_counts = [np.sum(dmg_cls[region_mask] == cls_indxex) for cls_indxex in classes]
                max_label = np.argmax(label_counts) + 1
                dmg_cls[region_mask] = max_label

            # print(region_label, area, solidity)

    dmg_cls[dmg_cls == 0] = 1  # Fill remaining with damage type 1 (no damage)
    return loc_cls.astype(np.uint8), dmg_cls.astype(np.uint8)


def make_predictions_most_severe(damage_probs: np.ndarray, min_size=32, assign_severe=True):
    loc_pred = np.stack((damage_probs[0, ...], np.sum(damage_probs[1:, ...], axis=0)))
    loc_cls = np.argmax(loc_pred, axis=0)

    # After we have 'fixed' localization predictions, we must zero-out probabilities for damage probs
    damage_probs = damage_probs.copy()
    damage_probs[0, loc_cls > 0] = 0

    dmg_cls = np.argmax(damage_probs, axis=0)

    buildings = label(loc_cls)

    if min_size is not None:
        # If there are any objects at all
        if buildings.max() > 0:
            buildings = remove_small_objects(buildings, min_size=min_size)
            buildings, _, _ = relabel_sequential(buildings)
            loc_cls = buildings > 0
            dmg_cls[~loc_cls] = 0

    if assign_severe:
        building_props = regionprops(buildings)
        classes = np.arange(1, 5)
        for index, region in enumerate(building_props):
            region_label, area, solidity = region["label"], region["area"], region["solidity"]

            region_mask = buildings == region_label

            if area < 2048 or solidity > 0.75:
                label_counts = np.array([np.sum(dmg_cls[region_mask] == cls_indxex) for cls_indxex in classes])
                if label_counts.sum() == 0:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.imshow(buildings)
                    plt.show()

                    plt.figure()
                    plt.imshow(region_mask)
                    plt.show()

                    plt.figure()
                    plt.imshow(dmg_cls)
                    plt.show()

                    breakpoint()

                min_count = max(1, label_counts[label_counts > 0].mean() - 3 * label_counts[label_counts > 0].std())

                labels = classes[label_counts >= min_count]
                max_label = labels.max()

                if len(labels) > 1:
                    print(label_counts, min_count, labels, max_label)
                # label_counts > 0
                # max_label = np.argmax(label_counts) + 1
                dmg_cls[region_mask] = max_label

            # print(region_label, area, solidity)

    dmg_cls[dmg_cls == 0] = 1  # Fill remaining with damage type 1 (no damage)
    return loc_cls.astype(np.uint8), dmg_cls.astype(np.uint8)


def make_predictions_floodfill(damage_probs: np.ndarray):
    loc_pred = np.stack((damage_probs[0, ...], np.sum(damage_probs[1:, ...], axis=0)))
    loc_cls = np.argmax(loc_pred, axis=0)

    # After we have 'fixed' localization predictions, we must zero-out probabilities for damage probs
    damage_probs = damage_probs.copy()
    damage_probs[0, loc_cls > 0] = 0

    seed = np.argmax(damage_probs, axis=0)

    dist = cv2.distanceTransform((1 - loc_cls).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
    dist = np.clip(dist, a_min=0, a_max=255).astype(np.uint8)

    # plt.figure()
    # plt.imshow(dist)
    # plt.show()

    img = np.dstack([dist, dist, dist])

    dmg_cls = cv2.watershed(img, seed.astype(int))
    if not isinstance(dmg_cls, np.ndarray):
        dmg_cls = dmg_cls.get()
    dmg_cls[dmg_cls < 1] = 1

    return loc_cls.astype(np.uint8), dmg_cls.astype(np.uint8)


def make_predictions_floodfill_with_image(damage_probs: np.ndarray, image):
    loc_pred = np.stack((damage_probs[0, ...], np.sum(damage_probs[1:, ...], axis=0)))
    loc_cls = np.argmax(loc_pred, axis=0)

    # After we have 'fixed' localization predictions, we must zero-out probabilities for damage probs
    damage_probs = damage_probs.copy()
    damage_probs[0, loc_cls > 0] = 0

    seed = np.argmax(damage_probs, axis=0)

    dmg_cls = cv2.watershed(image, seed.astype(int))
    if not isinstance(dmg_cls, np.ndarray):
        dmg_cls = dmg_cls.get()
    dmg_cls[dmg_cls < 1] = 1

    return loc_cls.astype(np.uint8), dmg_cls.astype(np.uint8)


def make_predictions_dominant_v2(
    damage_probs: np.ndarray, min_size=32, assign_dominant=True, max_building_area=4096, min_solidity=0.9
):
    """
    Combines floodfill and dominant postprocessing
    :param damage_probs:
    :param min_size:
    :param assign_dominant:
    :param max_building_area:
    :param min_solidity:
    :return:
    """
    loc_pred = np.stack((damage_probs[0, ...], np.sum(damage_probs[1:, ...], axis=0)))
    loc_cls = np.argmax(loc_pred, axis=0)

    # After we have 'fixed' localization predictions, we must zero-out probabilities for damage probs
    damage_probs = damage_probs.copy()
    damage_probs[0, loc_cls > 0] = 0
    dmg_cls = np.argmax(damage_probs, axis=0)

    buildings = label(loc_cls)

    if min_size is not None:
        # If there are any objects at all
        if buildings.max() > 0:
            buildings = remove_small_objects(buildings, min_size=min_size)
            buildings, _, _ = relabel_sequential(buildings)
            loc_cls = buildings > 0
            dmg_cls[~loc_cls] = 0

    if assign_dominant:
        building_props = regionprops(buildings)
        classes = list(range(1, 5))
        for index, region in enumerate(building_props):
            region_label, area, solidity = region["label"], region["area"], region["solidity"]

            region_mask = buildings == region_label

            if area < max_building_area and solidity > min_solidity:
                label_counts = [np.sum(dmg_cls[region_mask] == cls_indxex) for cls_indxex in classes]
                max_label = np.argmax(label_counts) + 1
                dmg_cls[region_mask] = max_label

            # print(region_label, area, solidity)

    dmg_cls[dmg_cls == 0] = 1  # Fill remaining with damage type 1 (no damage)
    return loc_cls.astype(np.uint8), dmg_cls.astype(np.uint8)
