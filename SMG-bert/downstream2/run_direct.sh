python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset bace --task_type classify --epochs 30 --represent_type 2D
#python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset Ames --task_type classify --epochs 30  --represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset BBBP --task_type classify --epochs 30 --represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset ESOL --task_type regression --epochs 30 --represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset FreeSolv --task_type regression --epochs 30 ---represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset Lipophilicity --task_type regression --epochs 30 --represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset LogS --task_type regression --epochs 30 --represent_type 2D


python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V0" --dataset bace --task_type classify --epochs 30 --use_3D --represent_type 3D
#python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V0" --dataset Ames --task_type classify --epochs 30 --use_3D --represent_type 3D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V0" --dataset BBBP --task_type classify --epochs 30 --use_3D --represent_type 3D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset ESOL --task_type regression --epochs 30 --represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset FreeSolv --task_type regression --epochs 30 ---represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset Lipophilicity --task_type regression --epochs 30 --represent_type 2D
python run.py --device cuda:1 --res_dir "/home/zjh/mr/res/downstream/V1" --dataset LogS --task_type regression --epochs 30 --represent_type 2D



