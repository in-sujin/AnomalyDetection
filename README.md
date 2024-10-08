# Anomaly Detection

#### DeepAnt
- paper : https://ieeexplore.ieee.org/document/8581424
- github : https://github.com/datacubeR/DeepAnt

#### MTAD-GAT
- paper : https://arxiv.org/abs/2009.02040
- github : https://github.com/ML4ITS/mtad-gat-pytorch

#### USAD
- paper : https://dl.acm.org/doi/10.1145/3394486.3403392
- github : https://github.com/manigalati/usad

---
#### TRAIN

```
python train.py 
```
- dataset : PRESTO / SWAT / MACHINE
- epochs : number of epochs
- batch_size : number of batch
- hidden_size : number of hidden
- window_size : window size for data loader


#### TEST ( SWAT / MACHINE )

```
python test.py
```
- dataset : PRESTO / SWAT / MACHINE
- epochs : number of epochs
- batch_size : number of batch
- hidden_size : number of hidden
- window_size : window size for data loader
- id : pretrained model id
