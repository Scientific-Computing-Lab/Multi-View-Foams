import numpy as np
import pandas as pd
import pickle
import os
import pdb
from config import data_dir, verbose


class DataExtract:
    def __init__(self,
                 data_path,
                 multiview=True,
                 examples_type='all',
                 no_yellow=False,
                 save_dir=None,
                 full_data_use=True):
        self.data_path = data_path
        self.multiview = multiview
        self.examples_type = examples_type
        self.no_yellow = no_yellow
        self.save_dir = save_dir
        self.full_data_use = full_data_use
        self.idxs_split = self.idxs_load()
        self.classes_id = {'red': 0,
                           'yellow': 0,
                           'green': 1}
        self.outer_group_names = np.unique(['-'.join(fname.split('-')[:3]) for fname in os.listdir(self.data_path)])
        self.labels = self.get_labels()
        self.inner_group_size = 1
        if self.examples_type == 'X10_both':
            self.inner_group_size = 2
        self.group_names, self.group_labels, self.group_labels_idx = self.outer_group_details(self.labels)
        if verbose > 1:
            print(f'number of groups: {len(self.group_names)}')

    def idxs_load(self):
        prefix = ''
        if not self.full_data_use:
            prefix = 'm2_'
        with open(os.path.join(data_dir, f'{prefix}idxs_split.pkl'), 'rb') as f:
            return pickle.load(f)

    def outer_group_details(self, labels):
        if self.multiview:
            if verbose > 2:
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

    def get_labels(self):
        images_labels = pd.read_excel(f'{data_dir}/image_labels.xlsx').iloc[:, 1:3]
        labels = pd.merge(pd.DataFrame({'model_name': self.outer_group_names}), images_labels, on=['model_name'], how='left')
        labels.dropna(inplace=True)
        if self.no_yellow:
            labels = labels.loc[labels['label'] != 'yellow']
        labels.reset_index(inplace=True)
        self.outer_group_names = self.outer_group_names[labels['index']]
        return labels

    def convert_to_real_idxs(self, group_train_idx, group_test_idx):
        group_train_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_train_idx
                           if len(self.labels[self.labels['index'] == idx].index) > 0]
        group_test_idx = [self.labels[self.labels['index'] == idx].index[0] for idx in group_test_idx
                          if len(self.labels[self.labels['index'] == idx].index) > 0]
        return np.array(group_train_idx), np.array(group_test_idx)

    def inner_group_idxs(self, group):
        group_idx = []
        for idx in group:
            group_idx += list(np.arange(idx * self.inner_group_size, idx * self.inner_group_size + self.inner_group_size))
        return group_idx

    def train_test_split(self):
        group_train_idx, group_test_idx = self.convert_to_real_idxs(self.idxs_split[0], self.idxs_split[1])
        if verbose > 2:
            print(f'group_train_idx: {group_train_idx} \n group_test_idx: {group_test_idx}')
        if self.examples_type == 'X10_both':
            if verbose > 2:
                print(f'inner_group_size: {self.inner_group_size}')
            return self.inner_group_idxs(group_train_idx), self.inner_group_idxs(group_test_idx)
        return group_train_idx, group_test_idx