import numpy as np
import os
import cv2
import pickle
from PIL import Image
from torchvision import transforms as tr
from datetime import datetime

from model1 import data_dir, images_dir


class DataPrep:
    def __init__(self, examples_type='all', augmentation={'split': True}):
        self.examples_type = examples_type
        self.augmentation = augmentation
        self.full_groups_only_dir = os.path.join(data_dir, 'full_groups_only')
        self.split_images_dir = os.path.join(data_dir, 'split_images')
        self.fnames_full_groups_only = os.listdir(self.full_groups_only_dir)
        self.examples_images = []
        self.group_names = []

        if not self.fnames_full_groups_only:
            self.set_group_names(dir=images_dir)
            self.full_groups_only()
        else:
            self.set_group_names(dir=self.full_groups_only_dir)
            for group_name in self.all_group_names:
                examples_images = self.examples_images_in_group(group_name=group_name, examples_type=self.examples_type)
                self.examples_images.append(examples_images)
            self.group_names = self.all_group_names

        if augmentation['split']:
            self.split_images()

    def set_group_names(self, dir):
        self.images_dir = dir
        fnames_images_dir = os.listdir(self.images_dir)
        self.all_group_names = list(np.unique(['-'.join(fname.split('-')[:3]) for fname in fnames_images_dir]))

    def examples_fnames_in_group(self, group_name, examples_type):
        examples = [filename for filename in os.listdir(self.full_groups_only_dir) if filename.startswith(group_name)]
        if examples_type == 'all':
            return examples
        return [example for example in examples if self.examples_type in [x.upper() for x in example.split('-')]]

    def examples_images_in_group(self, group_name, examples_type):
        examples_fnames = self.examples_fnames_in_group(group_name=group_name, examples_type=examples_type)
        return [cv2.imread(os.path.join(self.full_groups_only_dir, example_name)) for example_name in examples_fnames]

    def full_groups_only(self):
        for group_name in self.all_group_names:
            examples_images = self.examples_images_in_group(group_name=group_name, examples_type=self.examples_type)
            if len(examples_images) == 5:
                for example_idx, example in enumerate(examples_images):
                    cv2.imwrite(os.path.join(self.full_groups_only_dir, f'{group_name}-{example_idx}.png'), example)
                self.examples_images.append(examples_images)
                self.group_names.append(group_name)

    def split_images(self):
        for group_idx, group_examples in enumerate(self.examples_images):
            for example_idx, example in enumerate(group_examples):
                center_idx = int(example.shape[1] / 2)
                left_slice = example[:, :center_idx]
                right_slice = cv2.flip(example[:, center_idx:], 1)
                cv2.imwrite(os.path.join(self.split_images_dir, f'{self.group_names[group_idx]}-{example_idx}-left.png'), left_slice)
                cv2.imwrite(os.path.join(self.split_images_dir, f'{self.group_names[group_idx]}-{example_idx}-right.png'), right_slice)

data_prep = DataPrep()