import torch
import numpy as np
import pandas as pd
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import matplotlib.pyplot as plt

from utils import plot_predictions, loss_plot, ts_plot
from deepant import AnomalyDetector, DataModule, TrafficDataset, DeepAnt

import sklearn.metrics
from sklearn.metrics import roc_curve,roc_auc_score

pl.seed_everything(42, workers=True)

def config_args():
    
    parser = argparse.ArgumentParser(description='DeepAnt')
    
    parser.add_argument('--dataset', type=str, default='SWAT', help='dataset name')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='number of batch')

    parser.add_argument('--window_size', type=int, default=10, help='window size for data loader')

    parser.add_argument('--id', type=str, default='', help='wpretrained model id')

    args = parser.parse_args()
    
    return args

def dataloader(dataset):
    if dataset == 'SWAT':
        df = pd.read_csv('./data/SWAT/test.csv')

        df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.strip(), format='%d/%m/%Y %I:%M:%S %p')
        df.set_index('Timestamp', inplace=True)

        labels = [1 if label == 'Attack' else 0 for label in df['attack']]

        df = df.drop('attack', axis=1)

        return df, labels

    elif dataset == 'MACHINE':
        df = pd.read_csv('./data/MACHINE/test.csv')
        df = df.reset_index(drop=True)

        labels = df['attack']

        df = df.drop('attack', axis=1)

        return df, labels 
        
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
    y_pred = [np.array(item[1]) for item in outputs]

    # threshold
    threshold = ROC(y_test, y_pred)

    y_pred_ = np.zeros(len(y_pred))
    y_pred_ = [1.0 if (i >= threshold) else 0.0 for i in y_pred]

    print(sklearn.metrics.classification_report(y_test, y_pred_))


if __name__ == '__main__':
    
    # set arguments
    args = config_args()

    # load data
    df, labels = dataloader(args.dataset) 
    dataset = TrafficDataset(df, args.window_size)
    
    target_idx = dataset.timestamp
    
    feature_dim = df.shape[1]

    checkpoint_path = f'results/{args.dataset}/{args.id}.ckpt'

    # build model
    model = DeepAnt(feature_dim, args.window_size)
    anomaly_detector = AnomalyDetector.load_from_checkpoint(checkpoint_path, model = model)

    data_module = DataModule(df, args.window_size)
    model_checkpoint = ModelCheckpoint(
        dirpath = 'results/' + args.dataset,
        save_last = True,
        save_top_k = 1,
        verbose = True,
        monitor = 'train_loss', 
        mode = 'min'
    )

    # testing  
    trainer = pl.Trainer(max_epochs = args.epochs, # ^^epochs
                         accelerator = "gpu",
                         devices = 1, 
                         callbacks = [model_checkpoint]
    )
    
    outputs = trainer.predict(anomaly_detector, data_module)

    # evaluating 
    cal_metrics(labels, outputs, args.window_size)