import numpy as np
import pandas as pd
import pickle
import os
import pdb
from load_config import data_dir


class DataExtract:
    def __init__(self,
                 data_path,
                 multiple_folders,
                 binary_classes,
                 multiview=True,
                 dmvcnn=False,
                 examples_type='X10',
                 no_yellow=False,
                 given_idxs=False,
                 save_dir=None,
                 train_dir=None,
                 test_dir=None,
                 test_size=0.2):
        self.data_path = data_path
        self.multiple_folders = multiple_folders
        self.binary_classes = binary_classes
        self.multiview = multiview
        self.dmvcnn = dmvcnn
        self.examples_type = examples_type
        self.no_yellow = no_yellow
        self.given_idxs = given_idxs
        self.save_dir = save_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test_size = test_size
        if binary_classes:
            self.classes_id = {'red': 0,
                               'yellow': 0,
                               'green': 1}
        else:
            self.classes_id = {'red': 0,
                               'yellow': 1,
                               'green': 2}
        self.outer_group_names = np.unique(['-'.join(fname.split('-')[:3]) for fname in os.listdir(self.data_path)])
        if self.multiple_folders:
            self.inner_examples_size = len(os.listdir(os.path.join(self.data_path, self.outer_group_names[0])))
            self.inner_group_size = int(self.inner_examples_size / 5)
            if self.examples_type == 'X10_both':
                self.inner_group_size = self.inner_group_size * 2
        else:
            if self.examples_type == 'X10_both':
                self.inner_group_size = 2
                if self.dmvcnn:
                    self.inner_group_size = 10
            else:
                self.inner_group_size = 1
                if self.dmvcnn:
                    self.inner_group_size = 5
        self.group_names, self.group_labels, self.group_labels_idx = self.get_group_details()

    def inner_group_details(self, labels):
        inner_group_names = []
        inner_group_labels = []
        inner_group_labels_idx = []
        group_names = np.array(labels['model_name'])
        for idx, group_name in enumerate(group_names):
            if self.multiview:
                cur_inner_group_names = list(np.unique(['-'.join(fname.split('-')[:4]) for fname in os.listdir(os.path.join(self.data_path, group_name))]))
            else:
                if self.examples_type == 'X10_both':
                    cur_inner_group_names = [fname for fname in os.listdir(os.path.join(self.data_path, group_name))
                                             if int(fname.split('-')[-1][int(self.multiple_folders)]) < 2]
                else:
                    cur_inner_group_names = [fname for fname in os.listdir(os.path.join(self.data_path, group_name))
                                             if fname.split('-')[3][int(self.multiple_folders)] == self.examples_type[-1]]
            inner_group_names += cur_inner_group_names
            inner_group_labels += [self.classes_id[labels[labels['model_name'] == group_name]['label'].values[0]]] * len(cur_inner_group_names)
            label_idx = labels.loc[labels['model_name'] == group_name].index[0]
            inner_group_labels_idx += list(np.arange(label_idx * self.inner_group_size, label_idx * self.inner_group_size + self.inner_group_size))
        return inner_group_names, np.array(inner_group_labels), np.array(inner_group_labels_idx)

    def outer_group_details(self, labels):
        if self.multiview == True:
            print(f'outer group names: {self.outer_group_names}')
            return self.outer_group_names, np.array([self.classes_id[label] for label in labels['label']]), np.array(labels.index)
        else:
            inner_group_names = []
            inner_group_labels = []
            inner_group_labels_idx = []
            group_names = np.array(labels['model_name'])
            for idx, group_name in enumerate(group_names):
                group_fnames = [fname for fname in os.listdir(self.data_path) if fname.startswith(group_name)]
                if self.examples_type == 'X10_both':
                    cur_inner_group_names = [fname for fname in group_fnames if int(fname.split('-')[-1][0]) < 2]
                else:
                    cur_inner_group_names = [fname for fname in group_fnames if fname.split('-')[3][0] == self.examples_type[-1]]
                inner_group_names += cur_inner_group_names
                inner_group_labels += [self.classes_id[labels[labels['model_name'] == group_name]['label'].values[0]]] * len(cur_inner_group_names)
                label_idx = labels.loc[labels['model_name'] == group_name].index[0]
                inner_group_labels_idx += list(np.arange(label_idx * self.inner_group_size, label_idx * self.inner_group_size + self.inner_group_size))
            return inner_group_names, np.array(inner_group_labels), np.array(inner_group_labels_idx)

    def extract_examples(self, labels):
        if self.multiple_folders:
            return self.inner_group_details(labels)
        else:
            return self.outer_group_details(labels)

    def get_group_details(self):
        images_labels = pd.read_excel(f'{data_dir}/image_labels.xlsx').iloc[:, 1:3]
        labels = pd.merge(pd.DataFrame({'model_name': self.outer_group_names}), images_labels, on=['model_name'], how='left')
        labels.dropna(inplace=True)

        if self.no_yellow:
            labels = labels.loc[labels['label'] != 'yellow']
        labels.reset_index(inplace=True)
        self.labels = labels
        self.outer_group_names = self.outer_group_names[labels['index']]
        group_names, group_labels, group_labels_idx = self.extract_examples(labels)
        print(f'number of groups: {len(group_names)}')

        return group_names, group_labels, group_labels_idx

    def train_test_split_by_folder(self):
        group_test_names = np.unique(['-'.join(fname.split('-')[:3]) for fname in os.listdir(self.test_dir)])
        group_train_idx = []
        group_test_idx = []
        for idx, group_name in enumerate(self.outer_group_names):
            if group_name in group_test_names:
                group_test_idx.append(idx)
            else:
                group_train_idx.append(idx)
        return group_train_idx, group_test_idx

    def train_test_outer_split(self):
        groups_idx = np.arange(len(self.outer_group_names))
        if self.test_dir:
            return self.train_test_split_by_folder()
        np.random.shuffle(groups_idx)
        split_idx = int(self.test_size * len(self.outer_group_names))
        group_train_idx, group_test_idx = groups_idx[split_idx:], groups_idx[:split_idx]
        return group_train_idx, group_test_idx

    def convert_to_real_idxs(self, group_train_idx, group_test_idx):
        if self.no_yellow:
            group_train_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_train_idx
                               if self.labels[self.labels['index'] == idx].index != 0]
            group_test_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_test_idx
                              if self.labels[self.labels['index'] == idx].index != 0]
        else:
            group_train_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_train_idx]
            group_test_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_test_idx]
        return group_train_idx, group_test_idx

    def train_test_split(self):
        if self.given_idxs:
            with open(os.path.join(data_dir, 'idxs_split_additional_data.pkl'), 'rb') as f:
                idxs_split = pickle.load(f)
            group_train_idx, group_test_idx = self.convert_to_real_idxs(idxs_split[0], idxs_split[1])
        else:
            group_train_idx, group_test_idx = self.train_test_outer_split()
        print(f'group_train_idx: {group_train_idx} \n group_test_idx: {group_test_idx}')
        if self.multiple_folders:
            train_idx = []
            for idx in group_train_idx:
                train_idx += list(np.arange(idx * self.inner_group_size, idx * self.inner_group_size + self.inner_group_size))
            test_idx = []
            for idx in group_test_idx :
                test_idx += list(np.arange(idx * self.inner_group_size, idx * self.inner_group_size + self.inner_group_size))
        else:
            group_train_idx = np.array(group_train_idx)
            group_test_idx = np.array(group_test_idx)
            if self.examples_type == 'X10_both' or (self.dmvcnn==True and self.examples_type == 'X10_0') or (self.dmvcnn==True and self.examples_type == 'X10_1'):
                print(f'inner_group_size: {self.inner_group_size}')
                train_idx = []
                for idx in group_train_idx:
                    train_idx += list(
                        np.arange(idx * self.inner_group_size, idx * self.inner_group_size + self.inner_group_size))
                test_idx = []
                for idx in group_test_idx:
                    test_idx += list(
                        np.arange(idx * self.inner_group_size, idx * self.inner_group_size + self.inner_group_size))
                return train_idx, test_idx
            # last_idx = self.group_labels_idx[-1]
            # group_train_idx = group_train_idx[np.where(group_train_idx <= last_idx)[0]]
            # group_test_idx = group_test_idx[np.where(group_test_idx <= last_idx)[0]]
            return group_train_idx, group_test_idx
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        last_idx = self.group_labels_idx[-1]
        train_idx = train_idx[np.where(train_idx <= last_idx)[0]]
        test_idx = test_idx[np.where(test_idx <= last_idx)[0]]
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx