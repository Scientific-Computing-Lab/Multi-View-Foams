import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torchvision.transforms.functional as TF

from PIL import Image
from skimage.exposure import match_histograms
from config import full_groups_dir, preprocess_dir


def get_max_width_height(images):
    max_height = 0
    max_width = 0
    for img in images:
        height, width = img.shape
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
    return max_height, max_width


def rename_images(dir):
    idx = 0
    for fname in os.listdir(dir):
        if idx == 5:
            idx = 0
        group_name = '-'.join(fname.split('-')[:3])
        extension = fname.split('.')[-1]
        os.rename(os.path.join(dir, fname), os.path.join(dir, f'{group_name}-{idx}.{extension}'))
        idx += 1


def get_images(dir, matched_histograms=False):
    if matched_histograms:
        ref = cv2.imread(f'{full_groups_dir}/T483-2-5-0.png')
        return [cv2.cvtColor(match_histograms(cv2.imread(os.path.join(dir, filename)), ref, multichannel=True), cv2.COLOR_BGR2GRAY)
                if int(filename.split('-')[-1][0]) < 2
                else cv2.cvtColor(cv2.imread(os.path.join(dir, filename)), cv2.COLOR_BGR2GRAY)
                for filename in os.listdir(dir)], os.listdir(dir)
    return [cv2.cvtColor((cv2.imread(os.path.join(dir, filename))), cv2.COLOR_BGR2GRAY) for filename in os.listdir(dir)], os.listdir(dir)


def noise(image):
    image = np.array(image)
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.round(np.random.normal(mean, sigma, (row, col, ch)) * 10)
    gauss = gauss.reshape(row, col, ch)
    noisy = abs(image + gauss).astype(int)
    return noisy


def convert_to_bins(img, bins):
    img = np.array(img)
    jump = int(255/bins)
    for min_val in range(0, 255-2*jump, jump):
        max_val = min_val + jump
        img[(img >= min_val) & (img <= max_val)] = min_val
    min_val += jump
    img[(img >= min_val)] = min_val
    return img


def find_white_bar(img):
    return int(np.where(img[:, 100:500] == max(img[:, 0]))[0].mean())


def padding(img, max_height, max_width):
    white_bar = find_white_bar(img)
    new_img = np.zeros([max_height, max_width])
    img_width = img.shape[1]
    start = int(new_img.shape[0]/2)-white_bar
    end = start + img.shape[0]
    if end < max_height:
        new_img[start:end, :img_width] = img[:, :]
    else:
        new_img[start:max_height, :img_width] = img[:max_height-start, :]
    return new_img


def min_bounding_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center_coordinates = (int(x), int(y))
    radius = int(radius)
    return center_coordinates, radius


def find_contours(img):
    kernel = np.ones((5, 5), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=0)
    edged = cv2.Canny(img_dilate, 30, 200)
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def find_min_circle(img, verbose=0):
    contours, hierarchy = find_contours(img)
    if verbose > 0:
        cv2.drawContours(img, contours[0:], -1, (0, 255, 0), 3)
    center_coordinates, radius = min_bounding_circle(np.vstack(contours[0:]))
    return center_coordinates, radius


def bounding_square_crop(img):
    (x, y), radius = find_min_circle(img)
    return img[y-radius:y+radius, x-radius+1:x+radius]


def bounded_square(img, center_coordinates, radius, draw=False):
    x, y = center_coordinates
    print(center_coordinates)
    startpoint = (int(x - (radius / (2 ** 0.5))), int(y + (radius / (2 ** 0.5))))
    endpoint = (int(x + (radius / (2 ** 0.5))), int(y - (radius / (2 ** 0.5))))
    print(startpoint)
    print(endpoint)
    if draw:
        cv2.rectangle(img, startpoint, endpoint, (255, 0, 0), 2)
        plt.imshow(img, cmap='gray')
        plt.show()
    return img[endpoint[1]:startpoint[1], startpoint[0]:endpoint[0]]


def mask_circle(img, center_coordinates, radius=355, delta=0):
    mask = np.zeros(img.shape, dtype="uint8")
    cv2.circle(img, center_coordinates, radius+delta, (0, 255, 0), 2)
    cv2.circle(mask, center_coordinates, radius+delta, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


def center_circle(img):
    center_coordinates, radius = find_min_circle(img)
    img = np.roll(img, int(img.shape[1]/2) - center_coordinates[0], axis=1)
    img = np.roll(img, int(img.shape[0]/2) - center_coordinates[1], axis=0)
    return img, center_coordinates, radius


def detect_circle(img, radius, center_coordinates, verbose=0):
    img = mask_circle(img, center_coordinates=center_coordinates, radius=radius)
    center_coordinates, radius = find_min_circle(img, verbose)
    img = mask_circle(img, center_coordinates, radius)
    return img


def circle_permutation(img):
    # delta = 0  # 15, 35
    min_white_pixels = 10000000
    radii = list(range(295, 320, 4))
    for offset_x in range(-14, 60, 3):  # (-60, 60, 3)
        for offset_y in range(0, 100, 3):  # (-42, 60, 3)  (-100, 60, 3)
            for radius in radii:
                center_coordinates = (int(img.shape[0]/2) + offset_x, int(img.shape[1]/2) + offset_y)
                circle = mask_circle(img.copy(), center_coordinates=center_coordinates, radius=radius)
                white_pixels = len(np.where((circle >= 190))[0]) / len(np.where((circle < 190) & (circle != 0))[0])
                if white_pixels < min_white_pixels:
                    min_white_pixels = white_pixels
                    best_circle = circle
                    best_radius = radius
                    best_offset_x = offset_x
                    best_offset_y = offset_y
                    best_center_coordinates = center_coordinates
    print(f'{best_offset_x} {best_offset_y} {best_radius}\n')
    return best_circle


def multiple_boxes(img_original, filename, save_dir):
    (x, y), radius = find_min_circle(img_original)
    group_name = filename.split('.')[0]
    for rotation_num, angle in enumerate(range(0, 360, int(360 / 5))):
        img = np.array(TF.rotate(Image.fromarray(img_original.copy()), angle))
        img = bounded_square(img, (x, y), radius)
        cv2.imwrite(os.path.join(save_dir, f'{group_name}-{rotation_num}.png'), img)


def preprocess(images, filenames, save_dir, multi_rotations, upwards=True, profile=True):
    max_height, max_width = get_max_width_height(images)
    for idx, img in enumerate(images):
        if int(filenames[idx].split('-')[-1][0]) <= 1:
            if upwards:
                print(filenames[idx])
                img = bounding_square_crop(circle_permutation(img))
                if multi_rotations:
                    multiple_boxes(img, filenames[idx], save_dir)
                group_name = filenames[idx].split('.')[0]
                cv2.imwrite(os.path.join(save_dir, f'{group_name}.png'), img)
        elif profile:
            img = convert_to_bins(img, bins=10)
            img = padding(img, max_height, max_width)
            cv2.imwrite(os.path.join(save_dir, f'{filenames[idx]}'), img)


dir = full_groups_dir
save_dir = preprocess_dir
images, filenames = get_images(dir=dir, matched_histograms=False)
preprocess(images, filenames, save_dir=save_dir, multi_rotations=True, upwards=True, profile=False)
