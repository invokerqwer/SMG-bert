

import yaml
from easydict import EasyDict



from trainer import Pretrain_Trainer
from models import Pretrain_MR_BERT
config_file = """
bert:
    atom_vocab_size: 18
    embed_dim: 128
    ffn_dim: 512
    head: 4
    spec_endoder_layers: 2
    encoder_layers: 4
    p: 0.1
    use_3D: True
    use_1D: False
    use_adj_ssl: False
train:
    batch_size: 128
    num_workers: 4
    AdamW:
        lr: 5.0e-05
        beta1: 0.9
        beta2: 0.999
        eps: 1.0e-08
    epochs: 10
    use_gnn_adj: False
    mask_ratio: 0.2
dataset:
    # smiles_data_path: /home/zjh/mr/pretrain/dataset/sp2/smiles_test.pth
    # dist_data_path: /home/zjh/mr/pretrain/dataset/sp2/dist_test.pth
    # angle_data_path: /home/zjh/mr/pretrain/dataset/sp2/angle_test.pth
    # torsion_data_path: /home/zjh/mr/pretrain/dataset/sp2/torsion_test.pth
    
    # smiles_data_path: /home/zjh/mr/pretrain/dataset/sp2/smiles_test.pth
    # dist_data_path: 
    angle_data_path: 
    torsion_data_path: 
    
    
    smiles_data_path: /home/zjh/mr/pretrain/dataset/sp2/smiles.pth
    dist_data_path: /home/zjh/mr/pretrain/dataset/sp2/dist.pth
    # angle_data_path: /home/zjh/mr/pretrain/dataset/sp2/angle.pth
    # torsion_data_path: /home/zjh/mr/pretrain/dataset/sp2/torsion.pth
    mol_fix_len: 128
    num_workers: 0
    split: [8,1,1]
    pin_memory: True
device: cuda:2
seed: 42
res_dir: /home/zjh/mr/res/pretrain/V2
loss:
    loss_atom_1D: False
    loss_atom_2D: True
    loss_atom_3D: False
    adj_loss: False
    dist_loss: True
    angle_loss: False
    torsion_loss: False
    agg_method: weight
"""
config = EasyDict(yaml.safe_load(config_file))
model = Pretrain_MR_BERT(**config.bert)
trainer = Pretrain_Trainer(config,model)
trainer.train(config.train.epochs)