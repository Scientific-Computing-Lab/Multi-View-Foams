import pickle
from torchvision import transforms as tr
from model1_validation import ModelTrainer


new_run = True
SAVE = True
classes_names = ['bottle', 'cup']
surf_param = 300
img_shape = (300, 300)
test_size = 0.2

if new_run:
    model_trainer = ModelTrainer(classes_names=classes_names,
                                 surf_param=surf_param,
                                 img_shape=img_shape,
                                 test_size=test_size)
    if SAVE:
        model_trainer.save_feature_matrix()
        model_trainer.save_k_means()
else:
    with open(f'feature_matrix.pkl', 'rb') as f:
        feature_matrix = pickle.load(f)
    with open(f'k_means.pkl', 'rb') as f:
        k_means = pickle.load(f)
    centroids = k_means.cluster_centers_
    model_trainer = ModelTrainer(feature_matrix=feature_matrix,
                                 k_means=k_means,
                                 centroids=centroids,
                                 classes_names=classes_names,
                                 surf_param=surf_param,
                                 img_shape=img_shape,
                                 test_size=test_size)

print(f'The accuracy of Logistic Regression is {model_trainer.accuracy}%')
print('STOP')
