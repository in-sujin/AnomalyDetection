import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from utils import *
from usad import *

import torch.utils.data as data_utils

from datetime import datetime
import argparse


def config_args():
    parser = argparse.ArgumentParser(description='DeepAnt')
    
    parser.add_argument('--dataset', type=str, default='SWAT', help='dataset name')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch')
    parser.add_argument('--hidden_size', type=int, default=20, help='number of hidden')

    parser.add_argument('--window_size', type=int, default=100, help='window size for data loader')

    args = parser.parse_args()
    
    return args

def dataloader(dataset, window_size):
    if dataset == 'SWAT':
        df = pd.read_csv('./data/SWAT/train.csv', header=0, low_memory=False)
        window_data = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]
        
        return window_data
    
    elif dataset == 'MACHINE':
        df = pd.read_csv('./data/MACHINE/train.csv', header=0, low_memory=False)
        window_data = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]
        
        return window_data
        
    elif dataset == 'PRESTO':
        df = pd.read_csv('./data/PRESTO/train.csv', header=0, low_memory=False)
        df = df.reset_index(drop=True)

        window_data = df.values[np.arange(window_size)[None, :] + np.arange(df.shape[0] - window_size)[:, None]]

        return window_data
        
    else:
        print('dataset X')

    
if __name__ == '__main__':
    
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    # Set arguments
    args = config_args()
    
    # load data
    dataset = dataloader(args.dataset, args.window_size)
    print('train data : ', dataset.shape)
    
    w_size = dataset.shape[1] * dataset.shape[2]
    z_size = dataset.shape[1] * args.hidden_size
    
    train_dataset = dataset[:int(np.floor(.8 *  dataset.shape[0]))]
    val_dataset = dataset[int(np.floor(.8 *  dataset.shape[0])) : int(np.floor(dataset.shape[0]))]

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(train_dataset).float().view(([train_dataset.shape[0], w_size]))
    ) , batch_size=args.batch_size, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(val_dataset).float().view(([val_dataset.shape[0], w_size]))
    ) , batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # build model
    model = UsadModel(w_size, z_size)
    model = to_device(model, device)
    print(model)

    history = training(args.epochs, model, train_loader, val_loader)
        
    output_path = f'results/{args.dataset}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = f'{output_path}/{id}.pth'
    
    torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, save_path)