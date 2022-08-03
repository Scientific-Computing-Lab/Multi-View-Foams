import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from model2 import ObjectsDataset, MVCNN, CPU_Unpickler
from load_config import full_groups_dir, preprocess_dir, augmentation_dir, train_dir, test_dir, data_dir, models2_dir

full_groups_train_test_dir = os.path.join(data_dir, 'full_groups_train_test')
full_train_dir = os.path.join(full_groups_train_test_dir, 'train')
full_test_dir = os.path.join(full_groups_train_test_dir, 'test')


tests_dir = os.path.join(models2_dir, 'tests')
folders = ['bottom', 'top', 'top_bottom', 'multi_top_bottom', 'multi_all', 'multi_profiles']
# folders = ['multi_all', 'multi_profiles']
# Examples type:
# Multiview: all, X10, X20
# Seperated:  X10_0 (bottom), X10_1 (top), X10_both
examples_types = [['X10_0', 'X10_0'], ['X10_1', 'X10_1'], ['X10_both', 'X10_both'], ['X10', 'X10'], ['all', 'all'], ['X20', 'X20']]
# examples_types = [['all', 'all'], ['X20', 'X20']]

data_path = preprocess_dir
# data_path = os.path.join(data_dir, 'history/preprocess_most_updated')
dmvcnn = False
model_by_type = 'loss'   # loss / acc
augmentation = False
rotation = False
if data_path == augmentation_dir:
    multiple_folders = True
else:
    multiple_folders = False

binary_classes = True
given_idxs = True
multiview_arr = ['all', 'X10', 'X20']

num_classes = 3
if binary_classes:
    num_classes = 2
num_workers = 8


def settings(examples_type, j):
    multiview = False
    no_yellow = False
    if j == 1:
        no_yellow = True
    print('evaluating...')
    if examples_type in multiview_arr:
        multiview = True
    print(f'Examples type: {examples_type}')
    print(f'Multiview: {multiview}')
    print(f'No yellow: {no_yellow}\n')
    return multiview, no_yellow


def load_data(multiview, no_yellow):
    dataset = ObjectsDataset(data_path=data_path,
                             multiple_folders=multiple_folders,
                             binary_classes=binary_classes,
                             multiview=multiview,
                             dmvcnn=dmvcnn,
                             augmentation=augmentation,
                             rotation=rotation,
                             examples_type=examples_type,
                             no_yellow=no_yellow,
                             given_idxs=given_idxs,
                             save_dir=save_dir)
    train_indices, val_indices = dataset.dataExtract.train_test_split()
    print(np.array(dataset.group_names)[train_indices])
    print(np.array(dataset.group_names)[val_indices])
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=len(val_indices), sampler=val_sampler, num_workers=num_workers)
    return val_loader


def load_model():
    print(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MVCNN(num_classes=num_classes, pretrained=True)
    model.load_state_dict(torch.load(model_dir))
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
            reg, preds = torch.max(outputs, 1)

            reg = np.array(reg.cpu())
            labels = np.array(labels.cpu())
            probs = np.array(F.softmax(outputs, dim=1).cpu())
            preds = np.array(preds.cpu())
            print(f'preds: {preds}')
            fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
            # fpr, tpr, thresholds = roc_curve(labels.data, reg)
            print(f'thresholds: {thresholds}')
            # probs = F.softmax(outputs, dim=1)
            # print(probs)
            # outputs = np.max(np.array(outputs.cpu()), axis=1)
            print(f'labels: {labels} \n probs: {probs}')
            auc = roc_auc_score(labels, probs[:, 1])
            # auc = roc_auc_score(labels.data, reg)

            print('AUC: %.3f' % auc)
            with open(os.path.join(save_dir, f'auc_{int(auc * 100)}_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(auc, f)
            with open(os.path.join(save_dir, f'fpr_model_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(fpr, f)
            with open(os.path.join(save_dir, f'tpr_model_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(tpr, f)
            with open(os.path.join(save_dir, f'thresholds_model_by_{model_by_type}.pkl'), 'wb') as f:
                pickle.dump(thresholds, f)
    return fpr, tpr


def plot_graphs(save_dir):
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
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), bbox_inches='tight')
    plt.close()

    plt.plot(val_acc_history, label='val_acc')
    plt.plot(train_acc_history, label='train_acc')
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'accuracy_history.png'), bbox_inches='tight')
    plt.close()


for i, folder in enumerate(folders):
    folder_dir = os.path.join(tests_dir, f'{folder}')
    for j, examples_type in enumerate(examples_types[i]):
        save_dir = os.path.join(folder_dir, f'{examples_type}_{j}')
        model_name = [fname for fname in os.listdir(save_dir) if fname.startswith(f'model_by_{model_by_type}')][0]
        model_dir = os.path.join(save_dir, model_name)

        multiview, no_yellow = settings(examples_type=examples_type, j=j)
        val_loader = load_data(multiview=multiview, no_yellow=no_yellow)
        model, device = load_model()
        fpr, tpr = auc_calc(val_loader=val_loader, model=model)
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.savefig(os.path.join(save_dir, f'roc_model_by_{model_by_type}.png'))
        plt.close()
        plot_graphs(save_dir=save_dir)
