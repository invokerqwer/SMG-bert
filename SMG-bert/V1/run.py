import argparse
import os.path

import pandas as pd
import torch

from utils import seed_set
from models import MR_BERT
from easydict import EasyDict
import yaml
from trainer import Downstream_Trainer





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--res_dir', type=str)
    parser.add_argument('--task_type', type=str,default="regression")
    parser.add_argument('--pretrain_model_path', type=str, default="")
    parser.add_argument('--mode', type=str, default="2D")
    parser.add_argument("--nmr",action="store_true",help="")
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config.res_dir = args.res_dir
    config.device = args.device
    config.task_type = args.task_type
    config.pretrain_model_path = args.pretrain_model_path
    config.bert.nmr = args.nmr
    if args.mode == "3D":
        if args.task_type == "regression":
            dataset_dir = "/home/zjh/mr/downstream/dataset/regression_3D"
        else:
            dataset_dir = "/home/zjh/mr/downstream/dataset/classify"
    else:
        if args.task_type == "regression":
            dataset_dir = "/home/zjh/mr/downstream/dataset/regression"
        else:
            dataset_dir = "/home/zjh/mr/downstream/dataset/classify"
    seed_set(config.seed)
    for file_name in os.listdir(dataset_dir):
        config.dataset.dataset_path = os.path.join(dataset_dir, file_name)
        model = MR_BERT(**config.bert)


        trainer = Downstream_Trainer(config, model)

        trainer.train(config.train.epochs)






