import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import cv2

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import pdb

from load_config import data_dir, preprocess_dir
from model2 import MVCNN, DMVCNN

num_classes = 2
# model_dir = '/home/nadavsc/Desktop/projects/targets/model2/models/tests/dmvcnn/bottom/X10_0_1/model_by_loss_85.pt'
# model_dir = '/home/nadavsc/Desktop/projects/targets/model2/models/tests/runnings/top/X10_1_0/model_by_loss_81.pt'
model_dir = '/home/nadavsc/Desktop/projects/targets/model2/models/tests/additional_data/fc_in_features_128/top/X10_1_0/model_by_loss_72.pt'
dmvcnn = False


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dmvcnn:
        model = DMVCNN(num_classes=num_classes, pretrained=True)
    else:
        model = MVCNN(num_classes=num_classes, pretrained=True)
    model.load_state_dict(torch.load(model_dir))
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


def batch_predict(images, lime=True):
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0).unsqueeze(0)
    if lime:
        # batch = batch[np.newaxis, :, :, 0, :, :]
        batch = batch[0, :, 0, np.newaxis, :, :]
        batch = batch[:, np.newaxis, :, :, :]
    # batch = torch.stack([preprocess_transform(img)]).unsqueeze(0)
    print(batch.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device).to(torch.float32)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    print(probs)
    return probs.detach().cpu().numpy()

# def mvcnn_pred(img):
#     plug = torch.stack([transform(img)]).unsqueeze(0)
#     plug = plug.to(device)
#     pred = torch.nn.functional.softmax(model(plug)).argmax().item()
#     return pred


pill_transf = size_transf()
preprocess_transform = to_tensor()

# save_dir = os.path.join(data_dir, 'preprocess_defects')
# print('YADA')
# for filename in os.listdir(preprocess_dir):
#     img = pill_transf(get_image(os.path.join(save_dir, filename)))
#     fname = filename.split('.')[0]
#     cv2.imwrite(img, f'{fname}.png')

img_name = 'T489-1-8-1'
# img = get_image(os.path.join(preprocess_dir, f'{img_name}.png'))
img = get_image(os.path.join(data_dir, 'history/preprocess_most_updated/T489-1-8-1.png'))
img = Image.fromarray(np.array(img) / 255.0)
model, device = load_model()
torch.manual_seed(1)

# test_pred = batch_predict([np.array(pill_transf(img))], lime=False)
# print(test_pred)
# print(test_pred.squeeze().argmax())


explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)).astype('double'),
                                         batch_predict,  # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)  # number of images that will be sent to classification function

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
plt.imshow(mask)
plt.axis('off')
plt.savefig(f'/home/nadavsc/Desktop/projects/targets/model2/models/tests/{img_name}-mask.png', dpi=400)
plt.close()
plt.imshow(temp)
plt.axis('off')
plt.savefig(f'/home/nadavsc/Desktop/projects/targets/model2/models/tests/{img_name}-temp.png', dpi=400)
plt.close()
img_boundry1 = mark_boundaries(temp / 255.0, mask)
plt.imshow(img_boundry1)
plt.axis('off')
plt.savefig(f'/home/nadavsc/Desktop/projects/targets/model2/models/tests/{img_name}-mark_boundaries.png', dpi=400)
plt.close()