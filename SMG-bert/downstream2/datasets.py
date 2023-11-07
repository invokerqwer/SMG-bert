import time
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import Tokenizer
from utils import apply_atom_mask, Smiles_to_adjoin, gcn_norm_adj
from typing import *
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
class Downstream_Dataset(Dataset):
    def __init__(self,smile_data,label_data,pos_data=None,dist_data=None,dist_matrix_norm=False,angle_data=None,torsion_data = None,mol_fix_len=128,gnn_adj=True,mask_prob=0.2):
        super().__init__()
        self.smile_data = smile_data
        self.label_data = label_data
        self.pos_data = pos_data
        self.dist_data = dist_data
        self.angle_data = angle_data
        self.torsion_data = torsion_data
        self.mol_fix_len = mol_fix_len
        self.tokenizer = Tokenizer()
        self.dist_matrix_norm=dist_matrix_norm
        self.gnn_adj = gnn_adj
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.smile_data)

    def __getitem__(self, index):

        smiles = self.smile_data[index]
        label = self.label_data[index]
        atom_symbol_list, adjoin_matrix = Smiles_to_adjoin(smiles, explicit_hydrogens=True,size=self.mol_fix_len - 1)
        atom_symbol_list = self.tokenizer.add_special_atom(atom_symbol_list)

        atom_id_list = self.tokenizer.convert_atoms_to_ids(atom_symbol_list)
        # 最大是128
        raw_lth =  len(atom_id_list)
        adjoin_matrix_G = torch.zeros([self.mol_fix_len]*2)
        adjoin_matrix_G[0,:] = 1
        adjoin_matrix_G[:,0] = 1

        adjoin_matrix_G[1:len(adjoin_matrix)+1, 1:len(adjoin_matrix)+1] = adjoin_matrix
        atom_id_list = self.tokenizer.padding_to_size(atom_id_list,self.mol_fix_len)
        atom_id_list_raw = atom_id_list.copy()
        atom_id_list_mask,masked_flag = apply_atom_mask(atom_id_list,raw_lth=raw_lth,mask_prob=self.mask_prob)

        dist_i=dist_j=gnn_adj_G=dist_matrix_G=angle_i=angle_j=angle_k=angle=torsion_k=torsion_i=torsion_j=torsion_t =torsion= None



        if self.gnn_adj:
            gnn_adj_G = gcn_norm_adj(adjoin_matrix_G,add_self_loop=True)


        if self.dist_data:
            dist_matrix = self.dist_data[index][:self.mol_fix_len - 1,:self.mol_fix_len - 1]
            dist_i = torch.arange(raw_lth-1).repeat_interleave(raw_lth-1)
            dist_j = torch.arange(raw_lth-1).repeat(raw_lth-1)
            if self.dist_matrix_norm:
                dist_matrix = torch.nn.functional.normalize(dist_matrix,p=1)
            dist_matrix_G =  torch.zeros([self.mol_fix_len]*2)
            dist_matrix_G[1:len(dist_matrix)+1,1:len(dist_matrix)+1] = dist_matrix




        if self.angle_data:
            angle_i,angle_j,angle_k,angle = self.angle_data[index]
            angle_i, angle_j, angle_k = angle_i.long(),angle_j.long(),angle_k.long()

        if self.torsion_data:
            torsion_k,torsion_i,torsion_j,torsion_t,torsion = self.torsion_data[index]
            torsion_k, torsion_i, torsion_j, torsion_t = torsion_k.long(),torsion_i.long(),torsion_j.long(),torsion_t.long()
        return  atom_id_list_mask,masked_flag,raw_lth,dist_i,dist_j,atom_id_list_raw,adjoin_matrix_G,gnn_adj_G,dist_matrix_G,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion,label
    @staticmethod
    def convert_index(idx_tuple,mol_fix_len):
        # 第一个节点是global节点，需要跳过
        for i,idx in enumerate(idx_tuple):
            idx += 1
            idx += i * mol_fix_len
        return torch.concat(idx_tuple)

    def collate_fn(self,batch: List[Any]):
        atom_id_list_mask,masked_flag,raw_lth,dist_i,dist_j,atom_id_list_raw,adjoin_matrix_G,gnn_adj_G,dist_matrix_G,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion,label = tuple(zip(*batch))

        max_raw_lth = np.max(np.array(raw_lth))

        atom_id_list_mask =  torch.tensor(atom_id_list_mask)
        atom_id_list_raw =  torch.tensor(atom_id_list_raw)

        masked_flag = torch.tensor(masked_flag)
        label = torch.tensor(label)

        res = {"atom_id_raw": atom_id_list_raw[:, :max_raw_lth],
               'atom_id_mask': atom_id_list_mask[:, :max_raw_lth],
               'masked_flag': masked_flag[:, :max_raw_lth],
               "adjoin_matrix": torch.stack(adjoin_matrix_G)[:, :max_raw_lth, :max_raw_lth], "label": label,
               "dist_matrix": torch.tensor(0), "dist_i": torch.tensor(0), "dist_j": torch.tensor(0),
               "dist_mask": torch.tensor(0), "gnn_adj": torch.tensor(0), "angle_i": torch.tensor(0),
               "angle_j": torch.tensor(0), "angle_k": torch.tensor(0), "angle": torch.tensor(0),
               "torsion_k": torch.tensor(0), "torsion_i": torch.tensor(0), "torsion_j": torch.tensor(0),
               "torsion_t": torch.tensor(0), "torsion": torch.tensor(0)}

        if self.dist_data is not None:
            dist_matrix = torch.stack(dist_matrix_G)
            res["dist_matrix"] = dist_matrix
            res["dist_i"] =Downstream_Dataset.convert_index(dist_i,max_raw_lth)
            res["dist_j"] =Downstream_Dataset.convert_index(dist_j,max_raw_lth)
            dist_mask = torch.zeros_like(res["adjoin_matrix"])
            for arr,l in zip(dist_mask,raw_lth):
                arr[1:l,1:l] = torch.ones(l-1,l-1)
            res["dist_mask"] =  dist_mask.bool()
        if self.gnn_adj:
            gnn_adj = torch.stack(gnn_adj_G)[:,:max_raw_lth,:max_raw_lth]
            res["gnn_adj"] = gnn_adj
        if self.angle_data is not None:
            res["angle_i"] = Downstream_Dataset.convert_index(angle_i,max_raw_lth)
            res["angle_j"] = Downstream_Dataset.convert_index(angle_j,max_raw_lth)
            res["angle_k"] = Downstream_Dataset.convert_index(angle_k,max_raw_lth)
            res["angle"] = torch.concat(angle)
        if self.torsion_data is not None:
            res["torsion_k"] = Downstream_Dataset.convert_index(torsion_k,max_raw_lth)
            res["torsion_i"] = Downstream_Dataset.convert_index(torsion_i,max_raw_lth)
            res["torsion_j"] = Downstream_Dataset.convert_index(torsion_j,max_raw_lth)
            res["torsion_t"] = Downstream_Dataset.convert_index(torsion_t,max_raw_lth)
            res["torsion"] = torch.concat(torsion)

        return res




import time
def get_downstream_dataloader(config,logger,smiles_data_path,dist_data_path,angle_data_path,torsion_data_path,label_data_path):
    label_data = smiles_data = dist_data = torsion_data = angle_data =None

    if smiles_data_path:
        start = time.time()
        logger.info(f"start to load f{smiles_data_path}")
        smiles_data = torch.load(smiles_data_path)
        end = time.time()
        logger.info(f"end to load f{smiles_data_path},spend {end-start} ")

    if label_data_path:
        start = time.time()
        logger.info(f"start to load f{label_data_path}")
        label_data = torch.load(label_data_path)
        end = time.time()
        logger.info(f"end to load f{label_data_path},spend {end-start} ")

    if dist_data_path:
        start = time.time()
        logger.info(f"start to load f{dist_data_path}")
        dist_data = torch.load(dist_data_path)
        end = time.time()
        logger.info(f"end to load f{dist_data_path},spend {end-start} ")

    if torsion_data_path:
        start = time.time()
        logger.info(f"start to load f{torsion_data_path}")
        torsion_data = torch.load(torsion_data_path)
        end = time.time()
        logger.info(f"end to load f{torsion_data_path},spend {end-start} ")
    if angle_data_path:
        start = time.time()
        logger.info(f"start to load f{angle_data_path}")
        angle_data = torch.load(angle_data_path)
        end = time.time()
        logger.info(f"end to load f{angle_data_path},spend {end-start}")
    dataset = Downstream_Dataset(smiles_data,label_data=label_data,dist_data=dist_data,torsion_data=torsion_data,angle_data=angle_data,mol_fix_len=config.dataset.mol_fix_len,gnn_adj=config.train.use_gnn_adj)


    if config.dataset.split_type == "random":
        Num = len(dataset)
        tra_num, val_num = int(Num * 0.8), int(Num * 0.1)
        test_num = Num - (tra_num + val_num)
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [tra_num, val_num, test_num])
        logger.info(f"train_size {tra_num}, valid_size{val_num},test_size:{test_num}")
    else:
        train_inds,valid_inds,test_inds = split(dataset)
        train_dataset,valid_dataset,test_dataset = torch.utils.data.Subset(dataset, train_inds),torch.utils.data.Subset(dataset, valid_inds),torch.utils.data.Subset(dataset, test_inds)
        logger.info(f"train_size {len(train_dataset)}, valid_size{len(valid_dataset)},test_size:{len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                  collate_fn=dataset.collate_fn, shuffle=True,
                                  num_workers=config.train.num_workers,pin_memory=config.dataset.pin_memory)

    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size,
                                 collate_fn=dataset.collate_fn, num_workers=config.train.num_workers,pin_memory=config.dataset.pin_memory)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.train.batch_size,
                                  collate_fn=dataset.collate_fn,
                                  num_workers=config.train.num_workers,pin_memory=config.dataset.pin_memory)
    return train_dataloader, valid_dataloader, test_dataloader
def split(
    dataset: Dataset,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: Optional[int] = None,
    log_every_n: Optional[int] = 1000
) -> Tuple[List[int], List[int], List[int]]:

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(
                    scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

from typing import *
def generate_scaffolds(
                       dataset: Dataset,
                       log_every_n: int = 1000) -> List[List[int]]:
    """Returns all scaffolds from the dataset.
    Parameters
    ----------
    dataset: Dataset
        Dataset to be split.
    log_every_n: int, optional (default 1000)
        Controls the logger by dictating how often logger outputs
        will be produced.
    Returns
    -------
    scaffold_sets: List[List[int]]
        List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smile_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold,
             scaffold_set) in sorted(scaffolds.items(),
                                     key=lambda x: (len(x[1]), x[1][0]),
                                     reverse=True)
    ]
    return scaffold_sets

def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:


    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold
