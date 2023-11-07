


# 直接
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset bace --task_type classify --epochs 100 --represent_type 2D
#python run_seed1.py --device cuda:3--res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset Ames --task_type classify --epochs 100  --represent_type 2D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset BBBP --task_type classify --epochs 100 --represent_type 2D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset ESOL --task_type regression --epochs 100 --represent_type 2D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset FreeSolv --task_type regression --epochs 100 ---represent_type 2D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset Lipophilicity --task_type regression --epochs 100 --represent_type 2D
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V1" --dataset LogS --task_type regression --epochs 100 --represent_type 2D

python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset ESOL --task_type regression --epochs 100 --represent_type 3D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset FreeSolv --task_type regression --epochs 100 ---represent_type 3D
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset Lipophilicity --task_type regression --epochs 100 --represent_type 3D
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V0" --dataset LogS --task_type regression --epochs 100 --represent_type 3D



# 使用V6 引入3D 信息 没有loss_adj 不用gcn matrix
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V12" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V6/9.pth

python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset Ames --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset Lipophilicity --task_type regression --epochs 100 --use_3D --represent_type 3D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj
#python run_seed1.py --device cuda:3 --res_dir "/home/zjh/mr/res/downstream2_seed1/V10" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 2D --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --use_gnn_adj

