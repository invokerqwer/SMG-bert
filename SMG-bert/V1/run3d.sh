#!/bin/bash
python run_pretrain.py --config_path ./config_pretrain_3D.yml --device "cuda:1" --res_dir /home/zjh/mr/res_pretrain/V1_3D --type 3D --nmr --pos_info --lr=0.0001
python run_pretrain.py --config_path ./config_pretrain_3D.yml --device "cuda:1" --res_dir /home/zjh/mr/res_pretrain/V1_3D_noNMR --type 3D --pos_info --lr=0.0001
python run_pretrain.py --config_path ./config_pretrain_3D.yml --device "cuda:1" --res_dir /home/zjh/mr/res_pretrain/V1_3D_noPOS --type 3D --nmr --lr=0.0001
python run_pretrain.py --config_path ./config_pretrain_3D.yml --device "cuda:1" --res_dir /home/zjh/mr/res_pretrain/V1_3D_noBOTH --type 3D --lr=0.0001
python run_pretrain.py --config_path ./config_pretrain_3D.yml --device "cuda:1" --res_dir /home/zjh/mr/res_pretrain/V1_3D_lr10 --type 3D --nmr --pos_info --lr=10