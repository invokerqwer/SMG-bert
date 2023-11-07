# 直接
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset bace --task_type classify --epochs 100 --represent_type 2D
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset Ames --task_type classify --epochs 100  --represent_type 2D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset BBBP --task_type classify --epochs 100 --represent_type 2D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset ESOL --task_type regression --epochs 100 --represent_type 2D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset FreeSolv --task_type regression --epochs 100 ---represent_type 2D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset Lipophilicity --task_type regression --epochs 100 --represent_type 2D
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset LogS --task_type regression --epochs 100 --represent_type 2D

python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed02/V1" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V0" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset ESOL --task_type regression --epochs 100 --represent_type 2D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset FreeSolv --task_type regression --epochs 100 ---represent_type 2D
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset Lipophilicity --task_type regression --epochs 100 --represent_type 2D
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V1" --dataset LogS --task_type regression --epochs 100 --represent_type 2D


# 使用V3 引入3D 信息， 距离 + 重建
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V2" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth

python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V3" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth




# 使用 V5 引入3D 信息,所有的都用上，除了不用gcn matrix
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V4" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth



python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V5" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V5/9.pth


# 使用V9 引入3D 信息， 距离 + 重建 + angle
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V6" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth

python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V7" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V9/9.pth


# 使用v10 引入3D 信息， 距离 + 重建 + torsion
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V8" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth



python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V9" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V10/9.pth



# 使用V4 引入3D 信息,所有的都用上，包括gcn 也显示加入
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj




python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj


# 使用V6 引入3D 信息 没有loss_adj 不用gcn matrix
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V12" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth




python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V13" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V11" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V13" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V13" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V13" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V13" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
#python run_seed0.py --device cuda:2 --res_dir "/home/zjh/mr/res/downstream2_seed0/V10" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth