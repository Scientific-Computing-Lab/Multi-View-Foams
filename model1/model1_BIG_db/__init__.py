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


# PATHS
data_dir = os.path.join(dirname(dirname(abspath(__file__)), 'data_BIG_db'))
old_models = os.path.join(dirname(abspath(__file__)), 'old_models')


def normalize_data(x_train, x_test):
    delta = 1e-10
    m = x_train.mean(axis=0)
    std = x_train.std(axis=0) + delta
    x_train = (x_train - m) / std
    x_test = (x_test - m) / std
    return x_train, x_test


class ModelTrainer:
    def __init__(self, feature_matrix=None,
                 k_means=None,
                 centroids=None,
                 classes_names=[],
                 surf_param=300,
                 img_shape=(300, 300),
                 test_size=0.2):
        self.classes_names = classes_names
        self.surf_param = surf_param
        self.img_shape = img_shape
        self.examples_images = []
        self.group_names = []
        self.num_of_groups = {'bottle': 0,
                              'cup': 0}
        self.examples_images, self.group_names = self.get_examples_images()
        self.K = 200

        if feature_matrix is None:
            self.k_means, self.centroids = self.train_kmeans()
            self.feature_matrix = self.matrix_of_feature_vectors()
        else:
            self.feature_matrix = feature_matrix
            self.k_means = k_means
            self.centroids = centroids

        self._accuracy = 0
        self.test_size = test_size
        self.classes_id = {'bottle': 0,
                           'cup': 1}

        self.X, self.y = self.pre_process_for_train()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=self.test_size,
                                                                                shuffle=True,
                                                                                random_state=42)
        self.x_train, self.x_test = normalize_data(x_train=self.x_train,
                                                   x_test=self.x_test)
        self.train_logistic_regression()
        self.set_save_dir()
        # self.k_fold_accuracy()

    # @property
    # def num_of_groups(self):
    #     return self._num_of_groups

    @property
    def accuracy(self):
        y_pred = self.lr.predict(self.x_test)
        self._accuracy = round(accuracy_score(self.y_test, y_pred) * 100, 2)
        return self._accuracy

    def get_examples_images(self):
        examples_images = []
        group_names = []
        for class_name in self.classes_names:
            images_dir = os.path.join(data_dir, f'{class_name}/train')
            fnames = os.listdir(images_dir)
            all_group_names = list(np.unique([fname.split('.')[0] for fname in fnames]))
            self.num_of_groups[class_name] = len(all_group_names)
            for group_name in all_group_names:
                examples_fnames = self.examples_fnames(images_dir=images_dir, group_name=group_name)
                examples_images_in_group = [cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(images_dir, example_name)), cv2.COLOR_BGR2GRAY), self.img_shape)
                                   for example_name in examples_fnames]
                examples_images.append(examples_images_in_group)
                group_names.append(group_name)
        return examples_images, group_names

    def k_fold_accuracy(self):
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.X):
            x_train, x_test = normalize_data(self.X[train_index], self.X[test_index])
            y_train, y_test = self.y[train_index], self.y[test_index]
            lr = LogisticRegression()
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            print(f'accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}')

    def examples_fnames(self, images_dir, group_name):
        examples = [filename for filename in os.listdir(images_dir) if filename.startswith(group_name)]
        return examples

    def pre_process_for_train(self):
        X = self.feature_matrix
        y = np.append(np.zeros(self.num_of_groups['bottle']), np.ones(self.num_of_groups['cup']))
        return X, y

    def set_save_dir(self):
        save_folder_name = datetime.strftime(datetime.now(), f'%d%m%Y_%H%M {self.accuracy}_acc')
        self.save_dir = os.path.join(old_models, save_folder_name)

    def train_logistic_regression(self):
        self.lr = LogisticRegression()
        self.lr.fit(self.x_train, self.y_train)

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
        feature_matrix = np.empty((len(self.group_names), self.K))
        for group_idx in range(len(self.group_names)):
            feature_matrix[group_idx] = self.get_feature_vec(group_idx=group_idx)
        print('Shape of the feature matrix', feature_matrix.shape)
        return feature_matrix

    def save_feature_matrix(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        pickle.dump(self.feature_matrix, open(os.path.join(self.save_dir, f'feature_matrix.pkl'), 'wb'))

    def save_k_means(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        pickle.dump(self.k_means, open(os.path.join(self.save_dir, f'k_means.pkl'), 'wb'))
