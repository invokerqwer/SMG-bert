import time
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import Tokenizer
from utils import apply_atom_mask, Smiles_to_adjoin, gcn_norm_adj


class Pretrain_Dataset(Dataset):
    def __init__(self,smile_data,pos_data=None,dist_data=None,dist_matrix_norm=False,angle_data=None,torsion_data = None,mol_fix_len=128,gnn_adj=True,mask_prob=0.15):
        super().__init__()
        self.smile_data = smile_data
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
        return  raw_lth,dist_i,dist_j,atom_id_list_mask,atom_id_list_raw,masked_flag,adjoin_matrix_G,gnn_adj_G,dist_matrix_G,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion
    @staticmethod
    def convert_index(idx_tuple,mol_fix_len):
        # 第一个节点是global节点，需要跳过
        for i,idx in enumerate(idx_tuple):
            idx += 1
            idx += i * mol_fix_len
        return torch.concat(idx_tuple)

    def collate_fn(self, batch: List[Any]):
        raw_lth,dist_i,dist_j,atom_id_list_mask,atom_id_list_raw,masked_flag,adjoin_matrix_G,gnn_adj_G,dist_matrix_G,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = tuple(zip(*batch))

        max_raw_lth = np.max(np.array(raw_lth))
        atom_id_list_mask =  torch.tensor(atom_id_list_mask)
        atom_id_list_raw =  torch.tensor(atom_id_list_raw)

        masked_flag = torch.tensor(masked_flag)
        res = {'atom_id_mask': atom_id_list_mask[:,:max_raw_lth],
               "atom_id_raw":atom_id_list_raw[:,:max_raw_lth],
               "adjoin_matrix":torch.stack(adjoin_matrix_G)[:,:max_raw_lth,:max_raw_lth],
                'masked_flag': masked_flag[:,:max_raw_lth],
                }

        res["dist_matrix"] = res["dist_i"] = res["dist_j"] = res["dist_mask"]=res["gnn_adj"]=res["angle_i"]=res["angle_j"]=res["angle_k"]=res["angle"]=res["torsion_k"]=res["torsion_i"]=res["torsion_j"]=res["torsion_t"]=res["torsion"] = torch.tensor(0)

        if self.dist_data is not None:
            dist_matrix = torch.stack(dist_matrix_G)
            res["dist_matrix"] = dist_matrix
            res["dist_i"] =Pretrain_Dataset.convert_index(dist_i,max_raw_lth)
            res["dist_j"] =Pretrain_Dataset.convert_index(dist_j,max_raw_lth)
            dist_mask = torch.zeros_like(res["adjoin_matrix"])
            for arr,l in zip(dist_mask,raw_lth):
                arr[1:l,1:l] = torch.ones(l-1,l-1)
            res["dist_mask"] =  dist_mask.bool()
        if self.gnn_adj:
            gnn_adj = torch.stack(gnn_adj_G)[:,:max_raw_lth,:max_raw_lth]
            res["gnn_adj"] = gnn_adj
        if self.angle_data is not None:
            res["angle_i"] = Pretrain_Dataset.convert_index(angle_i,max_raw_lth)
            res["angle_j"] = Pretrain_Dataset.convert_index(angle_j,max_raw_lth)
            res["angle_k"] = Pretrain_Dataset.convert_index(angle_k,max_raw_lth)
            res["angle"] = torch.concat(angle)
        if self.torsion_data is not None:
            res["torsion_k"] = Pretrain_Dataset.convert_index(torsion_k,max_raw_lth)
            res["torsion_i"] = Pretrain_Dataset.convert_index(torsion_i,max_raw_lth)
            res["torsion_j"] = Pretrain_Dataset.convert_index(torsion_j,max_raw_lth)
            res["torsion_t"] = Pretrain_Dataset.convert_index(torsion_t,max_raw_lth)
            res["torsion"] = torch.concat(torsion)

        return res




import time
def get_pretrain_dataloader(config,logger,smiles_data_path,dist_data_path,angle_data_path,torsion_data_path):
    smiles_data = dist_data = torsion_data = angle_data =None

    if smiles_data_path:
        start = time.time()
        logger.info(f"start to load f{smiles_data_path}")
        smiles_data = torch.load(smiles_data_path)
        end = time.time()
        logger.info(f"end to load f{smiles_data_path},spend {end-start} ")
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
    dataset = Pretrain_Dataset(smiles_data,dist_data=dist_data,torsion_data=torsion_data,angle_data=angle_data,mol_fix_len=config.dataset.mol_fix_len,gnn_adj=config.train.use_gnn_adj,mask_prob=config.train.mask_ratio)
    Num = len(dataset)
    tra_num, val_num = int(Num * 0.8), int(Num * 0.1)
    test_num = Num - (tra_num + val_num)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [tra_num, val_num, test_num])
    logger.info(f"train_size {tra_num}, valid_size{val_num},test_size:{test_num}")
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                  collate_fn=dataset.collate_fn, shuffle=True,
                                  num_workers=config.train.num_workers,pin_memory=config.dataset.pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size,
                                 collate_fn=dataset.collate_fn, num_workers=config.train.num_workers,pin_memory=config.dataset.pin_memory)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.train.batch_size,
                                  collate_fn=dataset.collate_fn,
                                  num_workers=config.train.num_workers,pin_memory=config.dataset.pin_memory)
    return train_dataloader, valid_dataloader, test_dataloader

