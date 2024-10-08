import torch
import numpy as np
import pandas as pd
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse

from utils import plot_predictions, loss_plot, ts_plot
from deepant import AnomalyDetector, DataModule, TrafficDataset, DeepAnt



def config_args():
    parser = argparse.ArgumentParser(description='DeepAnt')
    
    parser.add_argument('--dataset', type=str, default='SWAT', help='dataset name')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch')

    parser.add_argument('--window_size', type=int, default=10, help='window size for data loader')

    args = parser.parse_args()
    
    return args

def dataloader(dataset):
    if dataset == 'SWAT':
        df = pd.read_csv('./data/SWAT/train.csv')

        df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.strip(), format='%d/%m/%Y %I:%M:%S %p')
        df.set_index('Timestamp', inplace=True)

        return df

    elif dataset == 'MACHINE':
        df = pd.read_csv('./data/MACHINE/train.csv')
        df = df.reset_index(drop=True)

        return df 
        
    elif dataset == 'PRESTO':
        df = pd.read_csv('./data/PRESTO/train.csv')
        df = df.reset_index(drop=True)

        return df 
        
    else:
        print('dataset X')

if __name__ == '__main__':
    
    # set arguments
    args = config_args()

    # load data
    df = dataloader(args.dataset) # ^^dataset
    dataset = TrafficDataset(df, args.window_size) # ^^window
    
    target_idx = dataset.timestamp 
    
    feature_dim = df.shape[1]

    # build model
    model = DeepAnt(feature_dim, args.window_size)
    anomaly_detector = AnomalyDetector(model)

    data_module = DataModule(df, args.window_size)
    model_checkpoint = ModelCheckpoint(
        dirpath = 'results/' + args.dataset,
        save_last = True,
        save_top_k = 1,
        verbose = True,
        monitor = 'train_loss', 
        mode = 'min'
    )

    checkpoint_name = 'DeepAnt-' + args.dataset
    model_checkpoint.CHECKPOINT_NAME_LAST = checkpoint_name

    summary(model)

    # training 
    trainer = pl.Trainer(max_epochs = args.epochs, # ^^epochs
                         accelerator = "gpu",
                         devices = 1, 
                         callbacks = [model_checkpoint]
    )
    trainer.fit(anomaly_detector, data_module)

    