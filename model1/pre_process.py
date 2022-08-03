import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from model1 import images_dir, split_images_dir, data_dir

full_groups_only_dir = os.path.join(data_dir, 'full_groups_only')

# GLOBALS
# Image Processing Parameters
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


# IMAGE PREPROCESSING FUNCTION (https://stackoverflow.com/questions/29313667/how-do-i-remove-the-background-from-this-kind-of-image)

def preprocess(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.show()

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    plt.imshow(edges, cmap='gray')
    plt.show()

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_area = cv2.contourArea(c)
        is_convex = cv2.isContourConvex(c)
        contour_info.append((c, is_convex, contour_area))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    # -- Create final image ---------------------------------------------------------------
    img[mask <= 100] = 0
    return img


# Preprocessing an example image
# filename = f'{images_dir}/T479-2-4-X10-AEROGEL-DOWN.tif'
filename = f'{full_groups_only_dir}/T486-1-9-2.png'
example = cv2.imread(filename)
center_idx = int(example.shape[0] / 2)
left_slice = example[:, :center_idx]
right_slice = cv2.flip(example[:, center_idx:], 1)
cv2.imwrite(os.path.join(data_dir, f'left.png'), left_slice)
cv2.imwrite(os.path.join(data_dir, f'right.png'), right_slice)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = preprocess(filename)
plt.imshow(example)
plt.show()