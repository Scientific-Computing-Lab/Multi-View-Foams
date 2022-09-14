import pickle
import os
from datetime import date
from torchvision import transforms as tr
from model1 import ModelTrainer, model_dir_config

from config import preprocess_dir


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


surf_param = 400  # 300, 400
k_centroids = 200  # 100, 200
img_shape = (224, 224)
full_data_use = True

save_dir = set_folders(k_centroids, full_data_use, no_yellow)
k_fold_size = 1000
test_size = 0.3

model_trainer = ModelTrainer(data_path=data_path,
                             binary_classes=True,
                             examples_type=examples_type,
                             full_groups_only=full_groups_only,
                             augmentation=augmentation,
                             surf_param=surf_param,
                             k_centroids=k_centroids,
                             img_shape=img_shape,
                             save_dir=save_dir,
                             k_fold_size=k_fold_size,
                             test_size=test_size)
model_trainer.save_feature_matrix()
model_trainer.save_k_means()

