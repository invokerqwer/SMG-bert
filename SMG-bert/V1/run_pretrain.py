import argparse
import os.path

import pandas as pd
import torch

from utils import seed_set
from models import Pretrain_MR_BERT,Pretrain_MR_BERT_3D
from easydict import EasyDict
import yaml
from trainer import Pretrain_Trainer
from trainer_3d import Pretrain_Trainer_3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--res_dir', type=str)
    parser.add_argument('--type', type=str,default="2D")
    parser.add_argument("--nmr",action="store_true",help="")
    parser.add_argument("--pos_info", action="store_true", help="")
    parser.add_argument('--lr',type=float,default=0.0)
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config.res_dir = args.res_dir
    config.device = args.device
    config.bert.nmr = args.nmr
    if not args.lr:
        config.bert.lr = args.lr
    seed_set(config.seed)
    if args.type == "3D":
        config.bert.pos_info = args.pos_info
        model = Pretrain_MR_BERT_3D(**config.bert)
        trainer = Pretrain_Trainer_3D(config, model)
    else:
        model = Pretrain_MR_BERT(**config.bert)
        trainer = Pretrain_Trainer(config, model)
    trainer.train(config.train.epochs)




