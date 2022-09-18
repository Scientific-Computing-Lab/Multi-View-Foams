import numpy as np
import pickle
import os

from datetime import date
from torchvision import transforms as tr
from model1 import ModelTrainer, model_dir_config

from config import preprocess_dir, models1_dir


def set_folders(k_centroids, full_data_use, no_yellow):
    cur_date = date.today().strftime("%d_%m_%Y")
    model_dir = model_dir_config(k_centroids, cur_date, full_data_use)
    save_dir = os.path.join(model_dir, f'{examples_type}_{int(no_yellow)}')
    try:
        os.mkdir(model_dir)
        os.mkdir(save_dir)
    except:
        print('Folder is already exists!')
    return save_dir


evaluation = True
data_path = preprocess_dir
examples_type = 'all'  # X10/X20/all
no_yellow = False
binary_classes = True
full_groups_only = True  # Includes only groups with all 5 examples
pipeline = tr.Compose([])
# pipeline = tr.Compose([tr.RandomHorizontalFlip(0.5),
#                        tr.ColorJitter(brightness=(0.8, 1.2))])
augmentation = {'split': False,
                'transforms': pipeline}
if examples_type != 'all':
    full_groups_only = False
full_data_use = True

save_dir = None

feature_matrix = None
k_means = None
centroids = None
surf_param = 400  # 300, 400
k_centroids = 200  # 100, 200
img_shape = (224, 224)
k_fold_size = 0
test_size = 0.3

if evaluation:
    folder_name = 'K_centroids_200_14_09_2022/all_0/model2_train_test_split'
    loading_dir = os.path.join(models1_dir, f'{folder_name}')
    with open(os.path.join(loading_dir, f'feature_matrix_{examples_type}.pkl'), 'rb') as f:
        feature_matrix = pickle.load(f)
    with open(os.path.join(loading_dir, f'k_means_{examples_type}.pkl'), 'rb') as f:
        k_means = pickle.load(f)
    centroids = k_means.cluster_centers_
else:
    save_dir = set_folders(k_centroids, full_data_use, no_yellow)


model_trainer = ModelTrainer(data_path=data_path,
                             no_yellow=no_yellow,
                             feature_matrix=feature_matrix,
                             k_means=k_means,
                             centroids=centroids,
                             examples_type=examples_type,
                             full_data_use=full_data_use,
                             augmentation=augmentation,
                             surf_param=surf_param,
                             k_centroids=k_centroids,
                             img_shape=img_shape,
                             save_dir=save_dir,
                             k_fold_size=k_fold_size,
                             test_size=test_size)
if not evaluation:
    model_trainer.save_feature_matrix()
    model_trainer.save_k_means()

