import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from utils import *
from usad import *

import torch.utils.data as data_utils

import sklearn.metrics
from sklearn.metrics import roc_curve,roc_auc_score

from datetime import datetime
import argparse

def config_args():
    parser = argparse.ArgumentParser(description='DeepAnt')
    
    parser.add_argument('--dataset', type=str, default='SWAT', help='dataset name')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch')
    parser.add_argument('--hidden_size', type=int, default=20, help='number of hidden')

    parser.add_argument('--window_size', type=int, default=100, help='window size for data loader')

    parser.add_argument('--id', type=str, default='', help='pretrained model id')

    args = parser.parse_args()
    
    return args

def dataloader(dataset, window_size):
    if dataset == 'SWaT':
        df = pd.read_csv('./data/SWaT/test.csv', header=0, low_memory=False)
        window_data = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]

        labels = pd.read_csv('./data/SWaT/labels.csv', header=0, low_memory=False)
        labels = labels['0'].to_list()

        
        return window_data, labels
    
    elif dataset == 'MACHINE':
        df = pd.read_csv('./data/MACHINE/test.csv', header=0, low_memory=False)
        window_data = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]

        labels = pd.read_csv('./data/MACHINE/labels.csv', header=0, low_memory=False)
        labels = labels['attack'].to_list()
        
        return window_data, labels
        
    elif dataset == 'presto':
        df = pd.read_csv('./data/PRESTO/test.csv', header=0, low_memory=False)
        df = df.reset_index(drop=True)
        
        window_data = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]

        lebels = pd.read_csv('./data/PRESTO/labels.csv', header=0, low_memory=False)
    
        
    else:
        print('dataset X')

def ROC(y_test,y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    return tr[idx]
    
def cal_metrics(labels, outputs, window_size):

    # original label
    windows_labels=[]
    for i in range(len(labels) - window_size):
        windows_labels.append(list(np.int_(labels[i : i + window_size])))

    y_test = [ 1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]
    
    # predicted output
    y_pred = np.concatenate([torch.stack(outputs[:-1]).flatten().detach().cpu().numpy(),
                              outputs[-1].flatten().detach().cpu().numpy()])

    # threshold
    threshold = ROC(y_test, y_pred)

    y_pred_ = np.zeros(len(y_pred))
    y_pred_ = [1.0 if (i >= threshold) else 0.0 for i in y_pred]

    print(sklearn.metrics.classification_report(y_test, y_pred_))


if __name__ == '__main__':

    # Set arguments
    args = config_args()
    
    # load data
    dataset, labels = dataloader(args.dataset, args.window_size)
    print('test data : ', dataset.shape)

    w_size = dataset.shape[1] * dataset.shape[2]
    z_size = dataset.shape[1] * args.hidden_size

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(dataset).float().view(([dataset.shape[0], w_size]))
    ) , batch_size=args.batch_size, shuffle=False, num_workers=0)

    # pretrained model 
    model_path = f'results/{args.dataset}/{args.id}.pth'
    checkpoint = torch.load(model_path)

    model = UsadModel(w_size, z_size)
    model = to_device(model, device)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])
    print(model)

    outputs = testing(model, test_loader, 0.5, 0.5)

    # evaluating 
    cal_metrics(labels, outputs, args.window_size)