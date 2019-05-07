import os
import sys

from skimage import io
from tqdm import tqdm
import numpy as np
from PIL import Image

from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from skimage.morphology import dilation, watershed, square, erosion
from skimage.measure import label, regionprops


TRAIN_PATH_IMGS = "../data/images/"
TRAIN_PATH_LABELS = "../data/labels/"

train_ids = next(os.walk(TRAIN_PATH_IMGS))[2]
traind_ids = sorted(train_ids)
label_ids = next(os.walk(TRAIN_PATH_LABELS))[2]
labeld_ids = sorted(label_ids)

print("Getting and reconstructing masks..")
sys.stdout.flush()

def create_contour(labels, mask_dilation=10, result_dilation=4, watershed_flag=True):
    mask = labels.copy()
    mask[mask > 0] = 1
    
    if watershed_flag:
        dilated = binary_dilation(mask, iterations=mask_dilation)
    else:
        dilated = binary_erosion(mask, iterations=mask_dilation)

    mask_w1 = watershed(dilated, labels, mask=dilated, watershed_line=True)
    mask_w1[mask_w1 > 0] = 1
    contours = dilated - mask_w1
    contours = binary_dilation(contours, iterations=result_dilation)
    return contours

for n, id_ in tqdm(enumerate(zip(traind_ids, labeld_ids)), total=len(train_ids)):
    path = TRAIN_PATH_IMGS + id_[0]
    label_path = TRAIN_PATH_LABELS + id_[1]
    img = io.imread(path)
    im = Image.fromarray(img)
    im.save("../data/TrainingData/images/" + str("%04d" % (n + 65)) + "_.tif")

    upper = 1
    lower = 0
    labels = io.imread(label_path)
    labels = np.where(labels > 0, 1, 0) * 255
    labels = labels.astype(dtype=np.uint8)

    watershed_labels = create_contour(label(labels))
    contour_labels = create_contour(labels, result_dilation=1, watershed_flag=False)

    mask_im = Image.fromarray(labels)
    mask_im.save("../data/TrainingData/labels/" + str("%04d" % (n + 65)) + "_.png")

    watershed_labels = watershed_labels * 255
    watershed_labels = watershed_labels.astype(np.uint8)
    watershed_mask = Image.fromarray(watershed_labels)
    watershed_mask.save("../data/TrainingData/watershed_labels/" + str("%04d" % (n + 65)) + "_.png")

    contour_labels = contour_labels * 255
    contour_labels = contour_labels.astype(np.uint8)
    contour_mask = Image.fromarray(contour_labels)
    contour_mask.save("../data/TrainingData/contour_labels/" + str("%04d" % (n + 65)) + "_.png")


