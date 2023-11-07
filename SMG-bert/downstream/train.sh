

# 无预训练
#python run.py --device cuda:1 --seed 42 --res_dir "./res/V1" --dataset FreeSolv --task_type regression --epochs 100 --use_3D --represent_type 3D --split_type scaffold
#python run.py --device cuda:1 --seed 42 --res_dir "./res/V1" --dataset ESOL --task_type regression --epochs 100 --use_3D --represent_type 3D --split_type scaffold

python run.py --device cuda:1 --seed 42 --res_dir "./res/V1" --dataset Lipophilicity --task_type classify --epochs 100 --use_3D --represent_type 3D --split_type random &
#python run.py --device cuda:1 --seed 42 --res_dir "./res/V1" --dataset chiral --task_type classify --epochs 100 --use_3D --represent_type 3D --split_type random &
python run.py --device cuda:1 --seed 42 --res_dir "./res/V1" --dataset bace --task_type classify --epochs 100 --use_3D --represent_type 3D --split_type scaffold
python run.py --device cuda:1 --seed 42 --res_dir "./res/V1" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --split_type scaffold

python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset bace --task_type classify --epochs 100  --represent_type 3D --use_3D --split_type scaffold --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset BBBP --task_type classify --epochs 100  --represent_type 3D --use_3D --split_type scaffold --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth

python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset chiral --task_type classify --epochs 100  --represent_type 3D --use_3D --split_type random --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth
python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset chiral --task_type classify --epochs 100  --represent_type 3D --use_3D --split_type random --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth
# 这个效果很好
python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset chiral --task_type classify --epochs 100  --represent_type 2D --use_3D --split_type random --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth
python run.py --device cuda:1 --seed 111 --res_dir "./res/V3" --dataset chiral --task_type classify --epochs 10  --represent_type 2D --use_3D --split_type random --pretrain_model_path /home/zjh/mr/res/pretrain/V4/9.pth --seed 48



python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset BBBP --task_type classify --epochs 100 --use_3D --represent_type 3D --split_type scaffold --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth

python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset LogS --task_type regression --epochs 100 --use_3D --represent_type 3D --split_type scaffold --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth



python run.py --device cuda:1 --seed 42 --res_dir "./res/V2" --dataset HIV --task_type classify --epochs 100 --use_3D --represent_type 3D --split_type scaffold --pretrain_model_path /home/zjh/mr/res/pretrain/V3/9.pth







