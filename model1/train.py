import pickle
from torchvision import transforms as tr
from model1 import ModelTrainer

from load_config import data_dir, augmentation_dir
# from test import ModelTrainer

new_run = True
SAVE = True
data_path = '/home/nadavsc/Desktop/projects/targets/data/augmentation'
multiple_folders = True
examples_type = 'all'  # X10/X20/all
binary_classes = True
full_groups_only = True  # Includes only groups with all 5 examples
pipeline = tr.Compose([])
# pipeline = tr.Compose([tr.RandomHorizontalFlip(0.5),
#                        tr.ColorJitter(brightness=(0.8, 1.2))])
augmentation = {'split': False,
                'transforms': pipeline}
if examples_type != 'all':
    full_groups_only = False
surf_param = 400 #300, 400
img_shape = (224, 224)
test_size = 0.2


if new_run:
    model_trainer = ModelTrainer(data_path=data_path,
                                 multiple_folders=multiple_folders,
                                 binary_classes=True,
                                 examples_type=examples_type,
                                 full_groups_only=full_groups_only,
                                 augmentation=augmentation,
                                 surf_param=surf_param,
                                 img_shape=img_shape,
                                 test_size=test_size)
    if SAVE:
        model_trainer.save_feature_matrix()
        model_trainer.save_k_means()
else:
    with open(f'feature_matrix_{examples_type}.pkl', 'rb') as f:
        feature_matrix = pickle.load(f)
    with open(f'k_means_{examples_type}.pkl', 'rb') as f:
        k_means = pickle.load(f)
    centroids = k_means.cluster_centers_
    model_trainer = ModelTrainer(data_path=data_path,
                                 multiple_folders=multiple_folders,
                                 binary_classes=True,
                                 feature_matrix=feature_matrix,
                                 k_means=k_means,
                                 centroids=centroids,
                                 full_groups_only=full_groups_only,
                                 augmentation=augmentation,
                                 surf_param=surf_param,
                                 img_shape=img_shape,
                                 test_size=test_size)
