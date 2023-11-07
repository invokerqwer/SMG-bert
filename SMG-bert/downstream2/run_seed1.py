import argparse
import os

import torch
import yaml
from easydict import EasyDict

from trainer2 import Downstream_Trainer
from models import Downstream_MR_BERT
from utils import seed_set

config_file = """
pretrain_model_path: 
bert:
    atom_vocab_size: 18
    embed_dim: 128
    ffn_dim: 512
    head: 4
    spec_endoder_layers: 2
    encoder_layers: 4
    p: 0.1
    use_3D: 
    use_1D: 
    use_adj_ssl: 
    task_type: 
    represent_type: 
train:
    batch_size: 
    num_workers: 0
    AdamW:
        lr: 5e-05
        beta1: 0.9
        beta2: 0.999
        eps: 1.0e-08
    epochs: 
    use_gnn_adj: 
    mask_ratio: 0.2
dataset:
    smiles_data_path: 
    dist_data_path:
    angle_data_path:
    torsion_data_path:
    label_data_path: 

    mol_fix_len: 128
    num_workers: 0
    split: [8,1,1]
    split_type: scaffold
    pin_memory: True
device: cuda:1
seed: 32
res_dir: 
loss:
    loss_atom_1D: 
    loss_atom_2D: 
    loss_atom_3D: 
    adj_loss: 
    dist_loss: 
    angle_loss: 
    torsion_loss: 
    agg_method: 
    alpha: 
"""


config = EasyDict(yaml.safe_load(config_file))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str)
    parser.add_argument('--res_dir', type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int,default=42)
    parser.add_argument('--pretrain_model_path', type=str, default="")

    parser.add_argument('--represent_type', type=str, default="2D")
    parser.add_argument('--task_type', type=str, default="regression")
    parser.add_argument("--use_3D", action="store_true", help="")
    parser.add_argument("--use_1D", action="store_true", help="")
    parser.add_argument("--use_adj_ssl", action="store_true", help="")

    parser.add_argument('--use_gnn_adj', action="store_true", help="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument("--loss_atom_1D", action="store_true", help="")
    parser.add_argument("--loss_atom_2D", action="store_true", help="")
    parser.add_argument("--loss_atom_3D", action="store_true", help="")
    parser.add_argument("--agg_method", action="store_true", default="weight", help="")
    parser.add_argument("--adj_loss", action="store_true", help="")
    parser.add_argument("--dist_loss", action="store_true", help="")
    parser.add_argument("--angle_loss", action="store_true", help="")
    parser.add_argument("--torsion_loss", action="store_true", help="")

    args = parser.parse_args()


    config.res_dir = args.res_dir
    config.device = args.device
    config.seed = args.seed
    config.pretrain_model_path = args.pretrain_model_path

    config.bert.task_type = args.task_type
    config.bert.represent_type = args.represent_type
    config.bert.use_3D = args.use_3D
    config.bert.use_1D = args.use_1D
    config.bert.use_adj_ssl = args.use_adj_ssl

    config.train.use_gnn_adj = args.use_gnn_adj
    config.train.batch_size = args.batch_size
    config.train.AdamW.lr = args.lr
    config.train.epochs = args.epochs

    config.loss.alpha = args.alpha
    config.loss.loss_atom_1D = args.loss_atom_1D
    config.loss.loss_atom_2D = args.loss_atom_2D
    config.loss.loss_atom_3D = args.loss_atom_3D

    config.loss.adj_loss = args.adj_loss
    config.loss.dist_loss = args.dist_loss
    config.loss.angle_loss = args.angle_loss
    config.loss.torsion_loss = args.torsion_loss
    config.loss.agg_method = args.agg_method

    if args.dataset == "bace":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/classify/bace/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/classify/bace/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/classify/bace/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/classify/bace/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/classify/bace/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "bace"))
    elif args.dataset == "Ames":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/classify/Ames/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/classify/Ames/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/classify/Ames/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/classify/Ames/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/classify/Ames/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "Ames"))
    elif args.dataset == "BBBP":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/classify/BBBP/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/classify/BBBP/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/classify/BBBP/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/classify/BBBP/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/classify/BBBP/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "BBBP"))
    elif args.dataset == "ESOL":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/regression/ESOL/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/regression/ESOL/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/regression/ESOL/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/regression/ESOL/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/regression/ESOL/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "ESOL"))
    elif args.dataset == "FreeSolv":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/regression/FreeSolv/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/regression/FreeSolv/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/regression/FreeSolv/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/regression/FreeSolv/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/regression/FreeSolv/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "FreeSolv"))
    elif args.dataset == "Lipophilicity":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/regression/Lipophilicity/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/regression/Lipophilicity/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/regression/Lipophilicity/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/regression/Lipophilicity/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/regression/Lipophilicity/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "Lipophilicity"))
    elif args.dataset == "LogS":
        config.dataset.smiles_data_path = "/home/zjh/mr/downstream/dataset/regression/LogS/smiles.pth"
        config.dataset.dist_data_path = "/home/zjh/mr/downstream/dataset/regression/LogS/dist.pth"
        config.dataset.angle_data_path = "/home/zjh/mr/downstream/dataset/regression/LogS/angle.pth"
        config.dataset.torsion_data_path = "/home/zjh/mr/downstream/dataset/regression/LogS/torsion.pth"
        config.dataset.label_data_path = "/home/zjh/mr/downstream/dataset/regression/LogS/label.pth"
        config.res_dir = str(os.path.join(config.res_dir, "LogS"))
    else:
        raise Exception("no such dataset")

    seed_set(config.seed)

    model = Downstream_MR_BERT(**config.bert)


    if config.pretrain_model_path:
        print(f"start to load pretrain model from {config.pretrain_model_path}")
        model.load_state_dict(torch.load(config.pretrain_model_path, map_location=config.device), strict=False)
    trainer = Downstream_Trainer(config, model)
    trainer.train(config.train.epochs)





