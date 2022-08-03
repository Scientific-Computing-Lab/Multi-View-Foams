import numpy as np
import pandas as pd
import os
import io
import random
import pdb
import torch
import pickle
import torch.nn as nn

from torch.utils.data import Dataset
from os.path import dirname, abspath
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# from pre_process import convert_to_bins, noise
from data_extract import DataExtract

# PATHS
project_path = dirname(dirname(abspath(__file__)))
data_dir = os.path.join(project_path, 'data')
images_dir = os.path.join(data_dir, 'preprocess')
old_models = os.path.join(dirname(abspath(__file__)), 'old_models')


def normalize(img):
    return np.float32(np.array(img) / 255)


def noise(image, grayscale):
    image = np.array(image)
    if grayscale:
        ch = 1
        row, col = image.shape
    else:
        row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.round(np.random.normal(mean, sigma, (row, col, ch)) * 10)
    gauss = gauss.reshape(row, col, ch)
    if grayscale:
        return np.uint8(abs(image + gauss[:, :, 0]))
    return np.uint8(abs(image + gauss))


def convert_to_bins(img, bins):
    img = np.array(img)
    jump = int(255/bins)
    for min_val in range(0, 255-2*jump, jump):
        max_val = min_val + jump
        img[(img >= min_val) & (img <= max_val)] = min_val
    min_val += jump
    img[(img >= min_val)] = min_val
    return img


def rotation_angles(filename, multiple_folders):
    if int(filename.split('-')[-1][int(multiple_folders)]) <= 1:
        # return [0, 90, 180, 270]
        return list(np.arange(0, 360, 30))
    return [-10, -5, 0, 5, 10]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = int(random.choice(self.angles))
        return TF.rotate(x, angle)


class ObjectsDataset(Dataset):
    def __init__(self, data_path,
                 multiple_folders=False,
                 binary_classes=False,
                 multiview=True,
                 dmvcnn = False,
                 examples_type='all',
                 no_yellow=False,
                 given_idxs=False,
                 save_dir=None,
                 train_dir=None,
                 test_dir=None,
                 augmentation=False,
                 rotation=False):
        self.data_path = data_path
        self.multiple_folders = multiple_folders
        self.examples_type = examples_type
        self.augmentation = augmentation
        self.rotation = rotation
        self.dataExtract = DataExtract(data_path=data_path,
                                       multiple_folders=multiple_folders,
                                       binary_classes=binary_classes,
                                       multiview=multiview,
                                       dmvcnn=dmvcnn,
                                       examples_type=examples_type,
                                       no_yellow=no_yellow,
                                       given_idxs=given_idxs,
                                       save_dir=save_dir,
                                       train_dir=train_dir,
                                       test_dir=test_dir)
        self.dmvcnn = dmvcnn
        self.group_names = self.dataExtract.group_names
        self.group_labels = self.dataExtract.group_labels
        self.group_labels_idx = self.dataExtract.group_labels_idx
        self.outer_group_names = self.dataExtract.outer_group_names
        print(f'{len(self.group_names)} {len(self.group_labels)} {len(self.group_labels_idx)} {len(self.outer_group_names)}')
        print(f'{self.group_names[0]}\n {self.group_names[1]} \n {self.group_names[2]} \n {self.group_names[3]}')

    def __len__(self):
        return len(self.group_names)

    def _transform(self, image, rotation_angles):
        resize = transforms.Compose([transforms.Resize((224, 224))])
        to_tensor = transforms.Compose([transforms.ToTensor()])
        image = resize(image)
        if not self.augmentation:
            image = Image.fromarray(np.array(image)/255.0)
            return to_tensor(image)

        noisy_image = Image.fromarray(noise(image, grayscale=True))
        color_transform = transforms.Compose([transforms.ColorJitter([0.8, 1.5], [0.8, 1.2], [0.8, 1.2], [-0.1, 0.1])])
        if self.rotation:
            image_rotation = MyRotationTransform(angles=rotation_angles)
            return to_tensor(normalize(image_rotation(color_transform(noisy_image))))
        else:
            return to_tensor(normalize(color_transform(noisy_image)))
            print('YADA')

    def stack_group(self, group_fnames, item_path):
        if self.examples_type == 'X10':
            return torch.stack([self._transform(Image.open(os.path.join(item_path, fname)).convert('L'), rotation_angles(fname, self.multiple_folders))
                                for fname in group_fnames if int(fname.split('-')[3][int(self.multiple_folders)]) < 2])
        if self.examples_type == 'X20':
            return torch.stack([self._transform(Image.open(os.path.join(item_path, fname)).convert('L'), rotation_angles(fname, self.multiple_folders))
                                for fname in group_fnames if int(fname.split('-')[3][int(self.multiple_folders)]) > 1])
        return torch.stack([self._transform(Image.open(os.path.join(item_path, fname)).convert('L'), rotation_angles(fname, self.multiple_folders))
                            for fname in group_fnames])

    def __getitem__(self, index):
        group_name = self.group_names[index]
        label = self.group_labels[index]
        # Get Images of the group
        item_path = self.data_path
        if self.multiple_folders:
            item_path = os.path.join(self.data_path, '-'.join(group_name.split('-')[:3]))
        group_fnames = [fname for fname in os.listdir(item_path) if fname.startswith(group_name)]
        if not self.dmvcnn:
            group_fnames = group_fnames[: 5]
        # print(f'{group_fnames}\n')
        group = self.stack_group(group_fnames, item_path)
        return group, label


# Return the number of learnable parameters for a given model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MVCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MVCNN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # fc_in_features = resnet.fc.in_features
        # fc_in_features = 64
        fc_in_features = 128
        # fc_in_features = 256
        # seq = list(resnet.children())[:-6]
        # seq.append(list(resnet.children())[:-5][-1][:-1])
        # self.features = nn.Sequential(*seq, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.features = nn.Sequential(*list(resnet.children())[:-4], nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # pdb.set_trace()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, num_classes)
        )

    def forward(self, inputs):  # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)

        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        outputs = self.classifier(pooled_views)
        return outputs


class DMVCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DMVCNN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        fc_in_features = 256
        # fc_in_features = 128
        self.features = nn.Sequential(*list(resnet.children())[:-3], nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, num_classes)
        )

    def forward(self, inputs):  # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        view_features_bottom = []
        view_features_top = []
        for idx, view_batch in enumerate(inputs):
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            if idx < 5:
                view_features_bottom.append(view_batch)
            else:
                view_features_top.append(view_batch)

        if inputs.shape[0] > 5:
            pooled_views_bottom, _ = torch.max(torch.stack(view_features_bottom), 0)
            pooled_views_top, _ = torch.max(torch.stack(view_features_top), 0)
            pooled_views, _ = torch.max(torch.stack([pooled_views_bottom, pooled_views_top]), 0)
        else:
            pooled_views, _ = torch.max(torch.stack(view_features_bottom), 0)
        outputs = self.classifier(pooled_views)
        return outputs
