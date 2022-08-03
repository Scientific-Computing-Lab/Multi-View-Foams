import numpy as np
import pandas as pd
import os
from os.path import dirname, abspath
import cv2
import pickle
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms as tr
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score

from data_extract import DataExtract

# PATHS
project_path = dirname(dirname(abspath(__file__)))
data_dir = os.path.join(project_path, 'data')
old_models = os.path.join(dirname(abspath(__file__)), 'old_models')


def normalize_data(x_train, x_test):
    delta = 1e-10
    m = x_train.mean(axis=0)
    std = x_train.std(axis=0) + delta
    x_train = (x_train - m) / std
    x_test = (x_test - m) / std
    return x_train, x_test


def logistic_regression_train(x_train, y_train):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    return lr


class ModelTrainer:
    def __init__(self,
                 data_path,
                 multiple_folders,
                 binary_classes,
                 feature_matrix=None,
                 k_means=None,
                 centroids=None,
                 examples_type='all',
                 full_groups_only=False,
                 augmentation={},
                 surf_param=300,
                 img_shape=(224, 224),
                 test_size=0.2):
        self.data_path = data_path
        self.multiple_folders = multiple_folders
        self.binary_classes = binary_classes
        self.examples_type = examples_type
        self.full_groups_only = full_groups_only
        self.augmentation = augmentation
        self.surf_param = surf_param
        self.img_shape = img_shape
        self.test_size = test_size

        self.dataExtract = DataExtract(data_path=data_path,
                                       multiple_folders=multiple_folders,
                                       binary_classes=binary_classes)
        self.group_names = self.dataExtract.group_names
        self.labels = self.dataExtract.group_labels
        self.labels_idx = self.dataExtract.group_labels_idx
        self.outer_group_names = self.dataExtract.outer_group_names

        self.examples_images = self.get_examples_images()
        self._num_of_groups = len(self.group_names)
        self.K = 200 #100, 200

        if feature_matrix is None:
            self.k_means, self.centroids = self.train_kmeans()
            self.feature_matrix = self.matrix_of_feature_vectors()
        else:
            self.feature_matrix = feature_matrix
            self.k_means = k_means
            self.centroids = centroids

        self.X, self.y = self.feature_matrix[self.labels_idx], self.labels
        self.set_save_dir()
        self.k_fold_accuracy()

    @property
    def num_of_groups(self):
        return self._num_of_groups

    def examples_fnames(self, item_path, group_name):
        filenames = [filename for filename in os.listdir(item_path) if filename.startswith(group_name)][:5]
        if self.examples_type == 'all':
            return filenames
        if self.examples_type == 'X10':
            return [filename for filename in filenames if int(filename.split('-')[-1][1]) < 2]
        return [filename for filename in filenames if int(filename.split('-')[-1][1]) > 1]

    def get_examples_images(self):
        examples_images = []
        for group_name in self.group_names:
            item_path = self.data_path
            if self.multiple_folders:
                item_path = os.path.join(self.data_path, '-'.join(group_name.split('-')[:3]))
            examples_fnames = self.examples_fnames(item_path=item_path, group_name=group_name)
            examples_images.append([cv2.resize(np.array(self.augmentation['transforms'](Image.open(os.path.join(item_path, example_name)))), self.img_shape)
                                    for example_name in examples_fnames])
        return examples_images

    def train_test_split(self):
        train_idx, test_idx = self.dataExtract.train_test_split()
        x_train, x_test, y_train, y_test = self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]
        x_train, x_test = normalize_data(x_train, x_test)
        return x_train, x_test, y_train, y_test

    def k_fold_accuracy(self):
        for k in range(5):
            x_train, x_test, y_train, y_test = self.train_test_split()
            lr = logistic_regression_train(x_train, y_train)
            y_pred = lr.predict(x_test)
            print(f'{k+1}) accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}')

    def set_save_dir(self):
        if self.full_groups_only:
            save_dir = os.path.join(old_models, 'full_groups_only')
        else:
            save_dir = os.path.join(old_models, 'all_groups')
        save_folder_name = datetime.strftime(datetime.now(), f'%d%m%Y_%H%M')
        self.save_dir = os.path.join(save_dir, self.examples_type, save_folder_name)

    def feature_extractor(self):
        surf = cv2.xfeatures2d.SURF_create(self.surf_param)  # create SURF feature extractor

        list_descriptors = []
        for group_examples in self.examples_images:
            for example in group_examples:
                kp, descriptors = surf.detectAndCompute(example, None)
                if descriptors is not None:
                    list_descriptors.append(descriptors)
        descriptor_matrix = np.vstack(list_descriptors)
        print('Shape of the Matrix of Descriptors', descriptor_matrix.shape)
        return descriptor_matrix

    def train_kmeans(self):
        descriptor_matrix = self.feature_extractor()
        k_means = KMeans(n_clusters=self.K)
        k_means.fit(descriptor_matrix)
        centroids = k_means.cluster_centers_
        return k_means, centroids

    def get_feature_vec(self, group_idx, surf_threshold=200):
        feature_vec = np.zeros(self.K)  # Intialize vector representation of visual words
        surf = cv2.xfeatures2d.SURF_create(surf_threshold)

        for example in self.examples_images[group_idx]:  # iterate over the image views of the car plug
            kp, descriptors = surf.detectAndCompute(example, None)  # extract the surf descriptors of the image
            if descriptors is not None:
                visual_words = self.k_means.predict(descriptors)  # find all the visual words in the image
                for visual_word in visual_words:  # find the occurence of each visual word
                    feature_vec[visual_word] += 1
        return feature_vec

    def matrix_of_feature_vectors(self):
        feature_matrix = np.empty((self.num_of_groups, self.K))
        for group_idx in range(len(self.group_names)):
            feature_matrix[group_idx] = self.get_feature_vec(group_idx=group_idx)
        print('Shape of the feature matrix', feature_matrix.shape)
        return feature_matrix

    def save_feature_matrix(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        pickle.dump(self.feature_matrix, open(os.path.join(self.save_dir, f'feature_matrix_{self.examples_type}.pkl'), 'wb'))

    def save_k_means(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        pickle.dump(self.k_means, open(os.path.join(self.save_dir, f'k_means_{self.examples_type}.pkl'), 'wb'))




