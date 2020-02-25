import pytest
import cv2
import numpy as np

import matplotlib.pyplot as plt

from xview.alignment import align_post_image_pyramid


def test_ecc():
    pre = cv2.imread("d:\\datasets\\xview2\\train\\images\\guatemala-volcano_00000002_pre_disaster.png")
    post = cv2.imread("d:\\datasets\\xview2\\train\\images\\guatemala-volcano_00000002_post_disaster.png")

    warpMatrix = np.zeros((3, 3), dtype=np.float32)
    warpMatrix[0, 0] = warpMatrix[1, 1] = warpMatrix[2, 2] = 1.0

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)

    retval = False

    try:
        retval, warpMatrix = cv2.findTransformECC(
            cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(post, cv2.COLOR_RGB2GRAY),
            warpMatrix,
            cv2.MOTION_HOMOGRAPHY,
            stop_criteria,
            None,
            5,
        )
        post_warped = cv2.warpPerspective(post, warpMatrix, dsize=(1024, 1024), flags=cv2.WARP_INVERSE_MAP)
    except:
        retval = False
        post_warped = post.copy()

    plt.figure()
    plt.imshow(pre)
    plt.show()

    plt.figure()
    plt.imshow(post)
    plt.show()

    plt.figure()
    plt.imshow(post_warped)
    plt.show()



def test_ecc_pyramid():
    pre = cv2.imread("c:\\datasets\\xview2\\train\\images\\guatemala-volcano_00000001_pre_disaster.png")
    post = cv2.imread("c:\\datasets\\xview2\\train\\images\\guatemala-volcano_00000001_post_disaster.png")
    post_warped = align_post_image_pyramid(pre, post)

    plt.figure()
    plt.imshow(pre)
    plt.show()

    plt.figure()
    plt.imshow(post)
    plt.show()

    plt.figure()
    plt.imshow(post_warped)
    plt.show()



def test_ecc_simple():
    pre = cv2.imread("pre.png")
    post = cv2.imread("post.png")

    warpMatrix = np.zeros((3, 3), dtype=np.float32)
    warpMatrix[0, 0] = warpMatrix[1, 1] = warpMatrix[2, 2] = 1.0

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.0001)

    retval = False

    try:
        retval, warpMatrix = cv2.findTransformECC(
            cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(post, cv2.COLOR_RGB2GRAY),
            warpMatrix,
            cv2.MOTION_HOMOGRAPHY,
            stop_criteria,
            None,
            5,
        )
        post_warped = cv2.warpPerspective(post, warpMatrix, dsize=(256, 256), flags=cv2.WARP_INVERSE_MAP)
    except:
        retval = False
        post_warped = post.copy()

    plt.figure()
    plt.imshow(pre)
    plt.show()

    plt.figure()
    plt.imshow(post)
    plt.show()

    plt.figure()
    plt.imshow(post_warped)
    plt.show()
