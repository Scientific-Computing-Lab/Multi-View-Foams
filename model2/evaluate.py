import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from model2 import ObjectsDataset, MVCNN, CPU_Unpickler, model_dir_config, verbose
from config import preprocess_dir, verbose


models_names = ['bottom', 'top', 'top_bottom', 'multi_top_bottom', 'multi_all', 'multi_profiles']
# Examples type:
# Seperated:  X10_0 (bottom), X10_1 (top), X10_both
multiview_arr = ['all', 'X10', 'X20']
examples_types = [['X10_0', 'X10_0'], ['X10_1', 'X10_1'], ['X10_both', 'X10_both'], ['X10', 'X10'], ['all', 'all'], ['X20', 'X20']]

# ---Model settings---
model_by_type = 'loss'   # loss / acc
fc_in_features = 128  # 64 / 128 / 256
num_workers = 8
# ---Model settings---

cur_date = '06_08_2022'  # date of the chosen model
data_path = preprocess_dir  # directory of the data set after pre-process
full_data_use = True  # if false use 20 examples less in train set


model_dir = model_dir_config(fc_in_features, cur_date, full_data_use)
if verbose > 0:
    print(model_dir)


def settings(examples_type, j):
    multiview = False
    no_yellow = False
    if j == 1:
        no_yellow = True
    if examples_type in multiview_arr:
        multiview = True
    if verbose > 0:
        print('evaluating...')
        print(f'Examples type: {examples_type}')
        print(f'Multiview: {multiview}')
        print(f'No yellow: {no_yellow}\n')
    return multiview, no_yellow


def load_data(multiview, no_yellow):
    dataset = ObjectsDataset(data_path=data_path,
                             multiview=multiview,
                             augmentation=False,
                             rotation=False,
                             examples_type=examples_type,
                             no_yellow=no_yellow,
                             save_dir=save_dir,
                             full_data_use=full_data_use)
    train_indices, val_indices = dataset.dataExtract.train_test_split()
    if verbose > 1:
        print(f'Val indices length: {len(val_indices)}')
        print(f'Train indices length: {len(train_indices)}')
    if verbose > 2:
        print(np.array(dataset.group_names)[train_indices])
        print(np.array(dataset.group_names)[val_indices])
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=len(val_indices), sampler=val_sampler, num_workers=num_workers)
    return val_loader


def load_model(model_type_dir):
    if verbose > 0:
        print(model_type_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MVCNN(fc_in_features=fc_in_features, pretrained=True)
    model.load_state_dict(torch.load(model_type_dir))
    model.eval()
    model.to(device)
    return model, device


def auc_calc(val_loader, model):
    torch.manual_seed(1)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = np.array(preds.cpu())
            outputs = np.array(F.softmax(outputs, dim=1).cpu())

            labels = np.array(labels.cpu())
            fpr, tpr, thresholds = roc_curve(labels, outputs[:, 1])
            auc = roc_auc_score(labels, outputs[:, 1])
            if verbose > 0:
                print('AUC: %.3f \n' % auc)
            if verbose > 2:
                print(f'preds: {preds}')
                print(f'thresholds: {thresholds}')
                print(f'labels: {labels} \n probs: {outputs} \n')

            with open(os.path.join(save_dir, f'auc_{int(auc * 100)}_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(auc, f)
            with open(os.path.join(save_dir, f'fpr_model_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(fpr, f)
            with open(os.path.join(save_dir, f'tpr_model_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(tpr, f)
            with open(os.path.join(save_dir, f'thresholds_model_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(thresholds, f)
    return fpr, tpr


def plot_graphs(save_dir, fpr=None, tpr=None):
    with open(os.path.join(save_dir, 'train_acc_history.pkl'), 'rb') as f:
        train_acc_history = CPU_Unpickler(f).load()
    with open(os.path.join(save_dir, 'train_loss_history.pkl'), 'rb') as f:
        train_loss_history = CPU_Unpickler(f).load()
    with open(os.path.join(save_dir, 'val_acc_history.pkl'), 'rb') as f:
        val_acc_history = CPU_Unpickler(f).load()
    with open(os.path.join(save_dir, 'val_loss_history.pkl'), 'rb') as f:
        val_loss_history = CPU_Unpickler(f).load()

    plt.plot(val_loss_history, label='val_loss')
    plt.plot(train_loss_history, label='train_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), bbox_inches='tight')
    plt.close()

    plt.plot(val_acc_history, label='val_acc')
    plt.plot(train_acc_history, label='train_acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'accuracy_history.png'), bbox_inches='tight')
    plt.close()

    if fpr is not None:
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.savefig(os.path.join(save_dir, f'roc_model_by_{model_by_type}.png'))
        plt.close()


for i, folder in enumerate(models_names):
    folder_dir = os.path.join(model_dir, f'{folder}')
    for j, examples_type in enumerate(examples_types[i]):
        save_dir = os.path.join(folder_dir, f'{examples_type}_{j}')
        model_name = [fname for fname in os.listdir(save_dir) if fname.startswith(f'model_by_{model_by_type}')][0]
        model_type_dir = os.path.join(save_dir, model_name)

        multiview, no_yellow = settings(examples_type=examples_type, j=j)
        val_loader = load_data(multiview=multiview, no_yellow=no_yellow)
        model, device = load_model(model_type_dir=model_type_dir)
        fpr, tpr = auc_calc(val_loader=val_loader, model=model)
        plot_graphs(save_dir=save_dir, fpr=fpr, tpr=tpr)
