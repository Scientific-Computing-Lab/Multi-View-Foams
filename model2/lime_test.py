import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import pdb

from model2 import MVCNN, model_dir_config, verbose
from config import preprocess_dir, verbose


# ---Model settings---
model_by_type = 'loss'   # loss / acc
fc_in_features = 128  # 64 / 128 / 256
num_workers = 8
# ---Model settings---

cur_date = '06_08_2022'
full_data_use = True
model_dir = model_dir_config(fc_in_features, cur_date, full_data_use)
if verbose > 0:
    print(model_dir)
multiple_imgs = True
images_dir = preprocess_dir
if not multiple_imgs:
    img_name = 'T483-2-8-1'

models_names = ['bottom', 'top', 'top_bottom']
# Examples type:
# Multiview: all, X10, X20
# Seperated:  X10_0 (bottom), X10_1 (top), X10_both
examples_types = [['X10_0', 'X10_0'], ['X10_1', 'X10_1'], ['X10_both', 'X10_both']]


def plot_graphs(img_name, params, save_dir):
    img_name = img_name.split('.')[0]
    for key, value in params.items():
        plt.imshow(value)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{img_name}-{key}.png'))
        plt.close()


def load_model(model_type_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MVCNN(fc_in_features=fc_in_features, pretrained=True)
    model.load_state_dict(torch.load(model_type_dir))
    model.eval()
    model.to(device)
    return model, device


def get_image(path, color=False):
    if color:
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    return Image.open(path).convert('L')


def size_transf():
    return transforms.Compose([transforms.Resize((224, 224))])


def to_tensor():
    return transforms.Compose([transforms.ToTensor()])


def batch_predict(images, verbose=0, lime=True):
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0).unsqueeze(0)
    if lime:
        batch = batch[0, :, 0, np.newaxis, :, :]
        batch = batch[:, np.newaxis, :, :, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device).to(torch.float32)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    if verbose > 1:
        print(f'batch shape: {batch.shape}')
    if verbose > 2:
        print(probs)
    return probs.detach().cpu().numpy()


def prediction_explain(img):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)).astype('double'),
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)  # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=8, hide_rest=False)
    return temp, mask


def single_prediction():
    test_pred = batch_predict([np.array(pill_transf(img))], lime=False)
    print(test_pred)
    print(test_pred.squeeze().argmax())


def run(img, img_name):
    global model, device
    for i, folder in enumerate(models_names):
        folder_dir = os.path.join(model_dir, f'{folder}')
        for j, examples_type in enumerate(examples_types[i]):
            save_dir = os.path.join(folder_dir, f'{examples_type}_{j}')
            model_name = [fname for fname in os.listdir(save_dir) if fname.startswith(f'model_by_{model_by_type}')][0]
            model_type_dir = os.path.join(save_dir, model_name)
            model, device = load_model(model_type_dir)
            torch.manual_seed(1)

            temp, mask = prediction_explain(img.copy())
            img_boundary = mark_boundaries(temp / 255.0, mask)
            params = {'mask': mask, 'temp': temp, 'marked_image': img_boundary}
            plot_graphs(img_name, params, os.path.join(save_dir, 'LIME'))


def lime_folders_open():
    for i, folder in enumerate(models_names):
        folder_dir = os.path.join(model_dir, f'{folder}')
        for j, examples_type in enumerate(examples_types[i]):
            try:
                os.mkdir(os.path.join(os.path.join(folder_dir, f'{examples_type}_{j}'), 'LIME'))
            except:
                if verbose > 0:
                    print(f'{folder} {examples_type}_{j}: LIME folder is already exists')


pill_transf = size_transf()
preprocess_transform = to_tensor()
if multiple_imgs:
    lime_folders_open()
    for img_name in os.listdir(images_dir):
        if int(img_name.split('.')[0][-1]) < 2:
            img = get_image(path=os.path.join(images_dir, img_name))
            img = Image.fromarray(np.array(img) / 255.0)
            run(img, img_name)
else:
    img = get_image(path=os.path.join(images_dir, f'{img_name}.png'))
    img = Image.fromarray(np.array(img) / 255.0)
    run(img, img_name)

