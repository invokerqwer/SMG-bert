import torch
from torch_sparse import SparseTensor
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import sys
import os
sys.path.append("/home/zjh/remote/mrbert/V3/")
from tokenizer import Tokenizer
from datasets import Pretrain_Dataset
from trainer import Pretrain_Trainer
from models import Pretrain_MR_BERT
config_file = """
bert:
    atom_vocab_size: 18
    embed_dim: 256
    ffn_dim: 512
    head: 4
    spec_endoder_layers: 2
    encoder_layers: 4
    p: 0.1
    pos_info:
train:
    batch_size: 256
    num_workers: 4
    AdamW:
        lr: 1.0e-04
        beta1: 0.9
        beta2: 0.999
        eps: 1.0e-08
    epochs: 20
dataset:
    # smiles_data_path: /home/zjh/mr/pretrain/dataset/sp2/smiles_test.pth
    # dist_data_path: /home/zjh/mr/pretrain/dataset/sp2/dist_test.pth
    # angle_data_path: /home/zjh/mr/pretrain/dataset/sp2/angle_test.pth
    # torsion_data_path: /home/zjh/mr/pretrain/dataset/sp2/torsion_test.pth
    smiles_data_path: /home/zjh/mr/pretrain/dataset/sp2/smiles.pth
    dist_data_path: /home/zjh/mr/pretrain/dataset/sp2/dist.pth
    angle_data_path: /home/zjh/mr/pretrain/dataset/sp2/angle.pth
    torsion_data_path: /home/zjh/mr/pretrain/dataset/sp2/torsion.pth
    mol_fix_len: 256
    num_workers: 0
    split: [8,1,1]
    pin_memory: True
device: cuda:1
seed: 42
res_dir: /home/zjh/mr/pretrain/V_std
loss:
    loss_atom_1D: True
    loss_atom_2D: True
    loss_atom_3D: True
    adj_loss: True
    dist_loss: True
    angle_loss: True
    torsion_loss: True
    agg_method: add
    
    
"""
config = EasyDict(yaml.safe_load(config_file))
model = Pretrain_MR_BERT(**config.bert)
trainer = Pretrain_Trainer(config,model)
trainer.train(50)