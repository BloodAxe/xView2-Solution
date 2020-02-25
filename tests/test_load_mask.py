import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from xview.dataset import read_mask
from xview.utils.inference_image_output import resize_mask_one_hot


def test_load_paletted():
    fname = "d:\\datasets\\xview2\\train\\masks\\hurricane-harvey_00000402_post_disaster.png"

    a = cv2.imread(fname)
    b = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    c = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)
    d = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)
    e = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
    f = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    g = np.array(Image.open(fname))

    print(a.shape, np.unique(a))
    print(b.shape, np.unique(b))
    print(c.shape, np.unique(c))
    print(d.shape, np.unique(d))
    print(e.shape, np.unique(e))
    print(f.shape, np.unique(f))
    print(g.shape, np.unique(g))



def test_mask_resize():
    fname = "d:\\datasets\\xview2\\train\\masks\\hurricane-harvey_00000402_post_disaster.png"
    mask = read_mask(fname)

    mask2 = cv2.resize(mask, (512,512), interpolation=cv2.INTER_NEAREST)
    mask3 = resize_mask_one_hot(mask, (512,512))

    cv2.imshow("Original", mask * 255)
    cv2.imshow("Nearest", mask2 * 255)
    cv2.imshow("Smart", mask3 * 255)

    cv2.waitKey(-1)

