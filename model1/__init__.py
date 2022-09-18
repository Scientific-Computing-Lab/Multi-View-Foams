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
from config import models1_dir

# PATHS
project_path = dirname(dirname(abspath(__file__)))
data_dir = os.path.join(project_path, 'data')
old_models = os.path.join(dirname(abspath(__file__)), 'old_models')


def accuracy(lr, x, y):
    return round(accuracy_score(y_true=y, y_pred=lr.predict(x)) * 100, 2)


def model_dir_config(K, cur_date, full_data_use=True):
    prefix = ''
    if not full_data_use:
        prefix = 'm2_'
    return os.path.join(models1_dir, f'{prefix}K_centroids_{K}_{cur_date}')


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
                 no_yellow,
                 feature_matrix=None,
                 k_means=None,
                 centroids=None,
                 examples_type='all',
                 full_data_use=True,
                 augmentation={},
                 surf_param=300,
                 k_centroids=200,
                 img_shape=(224, 224),
                 save_dir=None,
                 k_fold_size=20,
                 test_size=0.3):
        self.data_path = data_path
        self.examples_type = examples_type
        self.full_data_use = full_data_use
        self.augmentation = augmentation
        self.surf_param = surf_param
        self.img_shape = img_shape
        self.save_dir = save_dir
        self.k_fold_size = k_fold_size
        self.test_size = test_size

        self.dataExtract = DataExtract(data_path=data_path,
                                       multiview=True,
                                       examples_type=examples_type,
                                       no_yellow=no_yellow,
                                       save_dir=save_dir,
                                       full_data_use=True)
        self.group_names = self.dataExtract.group_names
        self.labels = self.dataExtract.group_labels
        self.labels_idx = self.dataExtract.group_labels_idx
        self.outer_group_names = self.dataExtract.outer_group_names

        self.examples_images = self.get_examples_images()
        self._n_groups = len(self.group_names)
        self.k_centroids = k_centroids

        if feature_matrix is None:
            self.k_means, self.centroids = self.train_kmeans()
            self.feature_matrix = self.matrix_of_feature_vectors()
        else:
            self.feature_matrix = feature_matrix
            self.k_means = k_means
            self.centroids = centroids

        self.X, self.y = self.feature_matrix[self.labels_idx], self.labels
        if self.k_fold_size == 0:
            x_train, x_test, y_train, y_test = self.data_prep(*self.dataExtract.train_test_split())
            lr = logistic_regression_train(x_train, y_train)
            acc = accuracy(lr, x_test, y_test)
            print(f'Accuracy: {acc}')
        else:
            self.k_fold_accuracy()

    @property
    def n_groups(self):
        return self._n_groups

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
            examples_fnames = self.examples_fnames(item_path=item_path, group_name=group_name)
            examples_images.append([cv2.resize(np.array(self.augmentation['transforms'](Image.open(os.path.join(item_path, example_name)))), self.img_shape)
                                    for example_name in examples_fnames])
        return examples_images

    def random_train_test_split(self):
        idxs = np.arange(0, self.n_groups)
        np.random.shuffle(idxs)

        idx_split = int(self.n_groups * self.test_size)
        return idxs[idx_split:], idxs[:idx_split]

    def data_prep(self, train_idx, test_idx):
        x_train, x_test, y_train, y_test = self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]
        x_train, x_test = normalize_data(x_train, x_test)
        return x_train, x_test, y_train, y_test

    def k_fold_accuracy(self):
        sum = 0
        np.random.seed(0)
        for k in range(self.k_fold_size):
            x_train, x_test, y_train, y_test = self.data_prep(*self.random_train_test_split())
            lr = logistic_regression_train(x_train, y_train)
            acc = accuracy(lr, x_test, y_test)
            sum += acc
            print(f'{k+1}) accuracy: {acc}')
        print(f'Average accuracy: {sum/self.k_fold_size}')

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
        k_means = KMeans(n_clusters=self.k_centroids)
        k_means.fit(descriptor_matrix)
        centroids = k_means.cluster_centers_
        return k_means, centroids

    def get_feature_vec(self, group_idx, surf_threshold=200):
        feature_vec = np.zeros(self.k_centroids)  # Intialize vector representation of visual words
        surf = cv2.xfeatures2d.SURF_create(surf_threshold)

        for example in self.examples_images[group_idx]:  # iterate over the image views of the car plug
            kp, descriptors = surf.detectAndCompute(example, None)  # extract the surf descriptors of the image
            if descriptors is not None:
                visual_words = self.k_means.predict(descriptors)  # find all the visual words in the image
                for visual_word in visual_words:  # find the occurence of each visual word
                    feature_vec[visual_word] += 1
        return feature_vec

    def matrix_of_feature_vectors(self):
        feature_matrix = np.empty((self.n_groups, self.k_centroids))
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




