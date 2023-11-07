python preprocess_downstream.py --file_path ./data/BBBP.csv --smiles_col_name smiles --label_col_name p_np --save_dir ./preprocess_data/BBBP &
python preprocess_downstream.py --file_path ./data/bace.csv --smiles_col_name mol --label_col_name Class --save_dir ./preprocess_data/bace &


python preprocess_downstream.py --file_path ./data/HIV.csv --smiles_col_name smiles --label_col_name HIV_active --save_dir ./preprocess_data/HIV &

python preprocess_downstream.py --file_path ./data/chiral.csv --smiles_col_name smiles --label_col_name "Chiral cliff" --save_dir ./preprocess_data/chiral &

python preprocess_downstream.py --file_path ./data/ESOL.csv --smiles_col_name smiles --label_col_name "measured log solubility in mols per litre" --save_dir ./preprocess_data/ESOL &
python preprocess_downstream.py --file_path ./data/FreeSolv.csv --smiles_col_name smiles --label_col_name expt --save_dir ./preprocess_data/FreeSolv &
python preprocess_downstream.py --file_path ./data/Lipophilicity.csv --smiles_col_name smiles --label_col_name exp --save_dir ./preprocess_data/Lipophilicity &
python preprocess_downstream.py --file_path ./data/tox21.csv --smiles_col_name smiles --label_col_name mol_id --save_dir ./preprocess_data/tox21 &

python preprocess_downstream.py --file_path ./data/Ames.csv --smiles_col_name Canonical_Smiles --label_col_name Activity --save_dir ./preprocess_data/Ames

python preprocess_downstream.py --file_path ./data/LogS.csv --smiles_col_name smiles --label_col_name LogS --save_dir ./preprocess_data/LogS



