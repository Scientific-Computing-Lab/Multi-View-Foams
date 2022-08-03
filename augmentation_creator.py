import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from PIL import Image
from load_config import full_groups_dir, preprocess_dir, augmentation_dir, data_dir


def noise(image):
    image = np.array(image)
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.round(np.random.normal(mean, sigma, (row, col)) * 10)
    gauss = gauss.reshape(row, col)
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


class AugmentationCreator:
    def __init__(self,
                 images,
                 filenames,
                 circles_ra=range(0, 360, 180),
                 profiles_ra=range(-0, 11, 10)):
        self.images = images
        self.filenames = filenames
        self.group_names = np.unique(['-'.join(fname.split('-')[:3]) for fname in self.filenames])
        self.color_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ColorJitter([1, 1.2], [1, 1], [1, 1], [0, 0])])
        self.flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5)])
        self.circles_ra = circles_ra # circles rotation angle
        self.profiles_ra = profiles_ra # profiles rotation angle

    def save_aug(self, aug_num, aug_dir, group_fnames, angles, images):
        for img_idx, (angle, img) in enumerate(zip(angles, images)):
            cv2.imwrite(os.path.join(aug_dir, f'{group_fnames[img_idx][0][:-5]}aug_{aug_num}-r{img_idx}_{angle}.png'), np.array(img))

    def transform_images(self, group_fnames, angles):
        images = [self.flip_transform(TF.rotate(self.color_transform(Image.fromarray(np.uint8(noise(self.images[group_fnames[group_idx][1]].copy())))), angle=angle))
                  if group_idx < 2 else self.flip_transform(TF.rotate(TF.to_pil_image(convert_to_bins(self.color_transform(Image.fromarray(np.uint8(noise(self.images[group_fnames[group_idx][1]].copy())))), 17)), angle=angle))
                  for group_idx, angle in enumerate(angles)]
        return images

    def folder_create(self, group_name):
        aug_dir = os.path.join(augmentation_dir, f'{group_name}')
        os.mkdir(aug_dir)
        return aug_dir

    def run(self):
        for group_idx, group_name in enumerate(self.group_names):
            print(f'{group_name}')
            print(f'Group number {group_idx + 1}')
            aug_dir = self.folder_create(group_name)
            aug_num = 0
            group_fnames = [[fname, idx] for idx, fname in enumerate(self.filenames) if fname.startswith(group_name)]
            for angle_1 in self.circles_ra:
                for angle_2 in self.circles_ra:
                    for angle_3 in self.profiles_ra:
                        for angle_4 in self.profiles_ra:
                            for angle_5 in self.profiles_ra:
                                angles = [angle_1, angle_2, angle_3, angle_4, angle_5]
                                images = self.transform_images(group_fnames, angles)
                                self.save_aug(aug_num, aug_dir, group_fnames, angles, images)
                                aug_num += 1

