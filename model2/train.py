import os
import time
import copy
from datetime import datetime
import pdb
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np

from model2 import ObjectsDataset, MVCNN, DMVCNN, count_parameters
from load_config import full_groups_dir, preprocess_dir, augmentation_dir, train_dir, test_dir, data_dir, models2_dir

full_groups_train_test_dir = os.path.join(data_dir, 'full_groups_train_test')
full_train_dir = os.path.join(full_groups_train_test_dir, 'train')
full_test_dir = os.path.join(full_groups_train_test_dir, 'test')


tests_dir = os.path.join(models2_dir, 'tests')
folders = ['bottom', 'top', 'top_bottom', 'multi_top_bottom', 'multi_all', 'multi_profiles']
folders = ['multi_all', 'multi_profiles']
# Examples type:
# Multiview: all, X10, X20
# Seperated:  X10_0 (bottom), X10_1 (top), X10_both
examples_types = [['X10_0', 'X10_0'], ['X10_1', 'X10_1'], ['X10_both', 'X10_both'], ['X10', 'X10'], ['all', 'all'], ['X20', 'X20']]
examples_types = [['all', 'all'], ['X20', 'X20']]


EPOCHS = 150
data_path = preprocess_dir
# data_path = os.path.join(data_dir, 'history/preprocess_new_data')
dmvcnn = False
if data_path == augmentation_dir:
    multiple_folders = True
    augmentation = False
    rotation = False
else:
    multiple_folders = False
    augmentation = True
    rotation = True
    if dmvcnn:
        rotation = False

binary_classes = True
given_idxs = True
multiview_arr = ['all', 'X10', 'X20']

num_classes = 3
if binary_classes:
    num_classes = 2
num_workers = 8


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts_loss = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Get model predictions
                    _, preds = torch.max(outputs, 1)
                    # pdb.set_trace()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.append(preds)
                all_labels.append(labels)

            epoch_loss = running_loss / len(dataloaders[phase].sampler.indices)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].sampler.indices)
            all_labels = torch.cat(all_labels, 0)


            print('{} Loss: {:.4f} - Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc_loss = epoch_acc
                best_model_wts_loss = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_by_acc_{int(best_acc * 100)}.pt'))
    model.load_state_dict(best_model_wts_loss)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_by_loss_{int(best_acc_loss*100)}.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc by loss: {:4f}'.format(best_acc_loss))

    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history, best_acc


for i, folder in enumerate(folders):
    folder_dir = os.path.join(tests_dir, f'{folder}')
    os.mkdir(folder_dir)
    for j, examples_type in enumerate(examples_types[i]):
        save_dir = os.path.join(folder_dir, f'{examples_type}_{j}')
        os.mkdir(save_dir)
        no_yellow = False
        multiview = False
        if j == 1:
            no_yellow = True
        print('training...')
        if examples_type in multiview_arr:
            multiview = True
        print(f'Examples type: {examples_type}')
        print(f'Multiview: {multiview}')
        print(f'No yellow: {no_yellow}\n')
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
        group_names = dataset.group_names
        y = dataset.group_labels
        outer_group_names = dataset.outer_group_names

        train_indices, val_indices = dataset.dataExtract.train_test_split()
        print(f'Val indices length: {len(val_indices)}')
        print(f'Train indices length: {len(train_indices)}')
        print(f'train group names: {np.array(dataset.group_names)[train_indices]}')
        print(f'test group names: {np.array(dataset.group_names)[val_indices]}')
        print(f'train group labels: {np.array(dataset.group_labels)[train_indices]}')
        print(f'test group labels: {np.array(dataset.group_labels)[val_indices]}')
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=4, sampler=val_sampler, num_workers=num_workers)
        data_loaders = {'train': train_loader, 'val': val_loader}

        if dmvcnn:
            model = DMVCNN(num_classes=num_classes, pretrained=True)
        else:
            model = MVCNN(num_classes=num_classes, pretrained=True)

        # DEFINE THE DEVICE
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(device)


        # UNFREEZE ALL THE WEIGHTS OF THE NETWORK
        for param in model.parameters():
            param.requires_grad = True
        # FINE-TUNE THE ENTIRE MODEL (I.E FEATURE EXTRACTOR + CLASSIFIER BLOCKS) USING A VERY SMALL LEARNING RATE
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)  # We use a smaller learning rate

        model, val_acc_history, val_loss_history, train_acc_history, train_loss_history, best_acc = train_model(model=model, dataloaders=data_loaders, criterion=criterion, optimizer=optimizer, num_epochs=EPOCHS)
        with open(os.path.join(save_dir, 'val_acc_history.pkl'), 'wb') as f:
            pickle.dump(val_acc_history, f)
        with open(os.path.join(save_dir, 'val_loss_history.pkl'), 'wb') as f:
            pickle.dump(val_loss_history, f)
        with open(os.path.join(save_dir, 'train_acc_history.pkl'), 'wb') as f:
            pickle.dump(train_acc_history, f)
        with open(os.path.join(save_dir, 'train_loss_history.pkl'), 'wb') as f:
            pickle.dump(train_loss_history, f)
        print('YADA')