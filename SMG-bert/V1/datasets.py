import bisect
import copy
from typing import Any, List, Dict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import Tokenizer
import numpy as np

from utils import Smiles_to_adjoin, pad_mol_nmr, apply_mask


class Masked_AtomNmr_Dataset(Dataset):
    def __init__(self,data = None, mol_fix_len=256):
        super().__init__()


        self.data = data
        self.mol_fix_len = mol_fix_len
        self.tokenizer = Tokenizer()


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):


        smi, nmr,labels = self.data[index]

        atom_symbol_list, adjoin_matrix = Smiles_to_adjoin(smi, explicit_hydrogens=True,size=self.mol_fix_len-1)

        # 预训练以及下游任务过滤NMR数值范围在[-800,1000],这里+801，变成[1,1801]
        # <pad> <global> 对应的NMR值不参与运算，因此用任意值即可
        # 而mask对应的NMR是需要参与计算的，使用0

        nmr = nmr + 801

        # 开头添加了<global>，(对应的nmr会被masked,因此设为任意值即可)
        atom_symbol_list_interim = self.tokenizer.add_special_atom(atom_symbol_list)
        nmr_interim = np.insert(nmr, 0,0)

        # 截断
        if len(atom_symbol_list) > self.mol_fix_len:
            atom_symbol_list_interim = atom_symbol_list_interim[:self.mol_fix_len]
            nmr_interim = nmr_interim[:self.mol_fix_len]

        mol_ids = np.array(self.tokenizer.convert_atoms_to_ids(atom_symbol_list_interim), np.int64)


        adjoin_matrix_G = np.ones([len(atom_symbol_list_interim),len(atom_symbol_list_interim)])
        adjoin_matrix_G[1:, 1:] = adjoin_matrix

        return  adjoin_matrix_G, mol_ids,nmr_interim,labels


    @staticmethod
    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        adjoin_matrix_G, mol_ids,nmr,labels = tuple(zip(*batch))


        batch_size = len(adjoin_matrix_G)
        # 该batch最大的图的shape,其余的图需要对齐到该shape
        shape = [batch_size] + [np.max([seq.shape for seq in adjoin_matrix_G])] * 2
        dtype = adjoin_matrix_G[0].dtype

        array = np.full(shape, 0, dtype=dtype)
        for arr, matrix in zip(array, adjoin_matrix_G):
            arrslice = tuple(slice(dim) for dim in matrix.shape)
            arr[arrslice] = matrix
        adjoin_matrix_G = torch.from_numpy(array)
        mol_ids = torch.from_numpy(pad_mol_nmr(mol_ids, 0)).long()
        nmr = torch.from_numpy(pad_mol_nmr(nmr, 0)).long()
        labels = torch.from_numpy(np.array(labels))
        return {'adjoin_matrix': adjoin_matrix_G,
                'mol_ids': mol_ids,
                'nmr': nmr,
                'labels': labels}

def get_dataloader(config,logger,pretrain=False,use_3D=False,pin_memory=False):
    data = torch.load(config.dataset.dataset_path)
    if pretrain:
        if use_3D:
            dataset = Pretrain_Masked_AtomNmr_Dataset_3D(data=data,mol_fix_len=config.dataset.mol_fix_len)
            collate_fn = Pretrain_Masked_AtomNmr_Dataset_3D.collate_fn
        else:
            dataset = Pretrain_Masked_AtomNmr_Dataset(data=data,mol_fix_len=config.dataset.mol_fix_len)
            collate_fn = Pretrain_Masked_AtomNmr_Dataset.collate_fn
    else:
        dataset = Masked_AtomNmr_Dataset(data=data, mol_fix_len=config.dataset.mol_fix_len)
        collate_fn = Masked_AtomNmr_Dataset.collate_fn

    # tra_num, val_num = int(len(dataset) * config.dataset.split[0] / sum(config.dataset.split)), int(len(dataset) * config.dataset.split[1] / sum(config.dataset.split))
    # test_num = len(dataset) - (tra_num + val_num)
    # train_dataset, valid_dataset, test_dataset = random_split(dataset, [tra_num, val_num, test_num])
    Num = len(dataset)
    tra_num, val_num = int(Num * 0.8), int(Num * 0.1)
    test_num = Num - (tra_num + val_num)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [tra_num, val_num, test_num])
    logger.info(f"train_size {tra_num}, valid_size{val_num},test_size:{test_num}")
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                  collate_fn=collate_fn, shuffle=True,
                                  num_workers=config.train.num_workers,pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size,
                                 collate_fn=collate_fn, num_workers=config.train.num_workers,pin_memory=pin_memory)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.train.batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=config.train.num_workers,pin_memory=pin_memory)
    return train_dataloader, valid_dataloader, test_dataloader

def get_stratify_dataloader(config,logger):
    def get_split_rank(data, stratify=10):
        lth = []
        for d in iter(data):
            lth.append(len(d[1]))
        split_rank = np.array(sorted(lth))[[int(len(lth) * rank / stratify) for rank in range(1, stratify)]]
        return split_rank

    def stratify_sampler(data, train_size, split_rank, ):
        lth = []
        for d in iter(data):
            lth.append(len(d[1]))
        rank_label = []
        for elem in lth:
            rank_label.append(bisect.bisect_left(split_rank, elem))
        return train_test_split(data, train_size=train_size, stratify=np.array(rank_label))

    data = torch.load(config.dataset.dataset_path)
    split_rank = get_split_rank(data)
    s1, s2 =  config.dataset.split[0] / sum(config.dataset.split), config.dataset.split[1] / (config.dataset.split[1]+config.dataset.split[2])
    train_data, other_data = stratify_sampler(data, s1, split_rank)
    valid_data, test_data = stratify_sampler(other_data,s2, split_rank)
    train_dataset = Masked_AtomNmr_Dataset(data=train_data, mol_fix_len=config.dataset.mol_fix_len)
    valid_dataset = Masked_AtomNmr_Dataset(data=valid_data, mol_fix_len=config.dataset.mol_fix_len)
    test_dataset = Masked_AtomNmr_Dataset(data=test_data, mol_fix_len=config.dataset.mol_fix_len)
    logger.info(f"train_size {len(train_dataset)}, valid_size{len(valid_dataset)},test_size:{len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                  collate_fn=Masked_AtomNmr_Dataset.collate_fn, shuffle=True,
                                  num_workers=config.train.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size,
                                 collate_fn=Masked_AtomNmr_Dataset.collate_fn, num_workers=config.train.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.train.batch_size,
                                  collate_fn=Masked_AtomNmr_Dataset.collate_fn,
                                  num_workers=config.train.num_workers)
    return train_dataloader, valid_dataloader, test_dataloader



class Pretrain_Masked_AtomNmr_Dataset(Dataset):
    def __init__(self,data = None, mol_fix_len=256):
        super().__init__()
        self.data = data
        self.mol_fix_len = mol_fix_len
        self.tokenizer = Tokenizer()


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):


        smi, nmr  = self.data[index]
        atom_symbol_list, adjoin_matrix = Smiles_to_adjoin(smi, explicit_hydrogens=True,size=self.mol_fix_len - 1)

        # 预训练以及下游任务过滤NMR数值范围在[-800,1000],这里+801，变成[1,1801]
        # <pad> <global> 对应的NMR值不参与运算，因此用任意值即可
        # 而mask对应的NMR是需要参与计算的，使用0

        nmr = nmr + 801

        # 开头添加了<global>，(对应的nmr会被masked,因此设为任意值即可)
        atom_symbol_list_interim = self.tokenizer.add_special_atom(atom_symbol_list)
        nmr_interim = np.insert(nmr, 0,0)

        # 截断
        if len(atom_symbol_list) > self.mol_fix_len:
            atom_symbol_list_interim = atom_symbol_list_interim[:self.mol_fix_len]
            nmr_interim = nmr_interim[:self.mol_fix_len]

        raw_nmr = copy.copy(nmr_interim)
        raw_atom_symbol_list = copy.copy(atom_symbol_list_interim)
        raw_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(raw_atom_symbol_list), np.int64)


        masked_atom_symbol_list,masked_nmr,masked_flag = apply_mask(atom_symbol_list_interim, nmr_interim,self.tokenizer)


        masked_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(masked_atom_symbol_list), np.int64)
        masked_flag =  np.array(masked_flag,np.int64)

        adjoin_matrix_G = np.ones([len(atom_symbol_list_interim),len(atom_symbol_list_interim)])
        adjoin_matrix_G[1:, 1:] = adjoin_matrix

        return  adjoin_matrix_G, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list,


    @staticmethod
    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        adjoin_matrix_G, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list, = tuple(zip(*batch))


        batch_size = len(adjoin_matrix_G)
        # 该batch最大的图的shape,其余的图需要对齐到该shape
        shape = [batch_size] + [np.max([seq.shape for seq in adjoin_matrix_G])] * 2
        dtype = adjoin_matrix_G[0].dtype

        array = np.full(shape, 0, dtype=dtype)
        for arr, matrix in zip(array, adjoin_matrix_G):
            arrslice = tuple(slice(dim) for dim in matrix.shape)
            arr[arrslice] = matrix
        adjoin_matrix_G = torch.from_numpy(array)

        # 将每个分子的nmr列表长度对齐（对齐到该batch中拥有最多原子的分子),
        # <padding>对应的nmr不会参与运算，设为任意值即可（[0,1801],这里设为0
        masked_mol_ids = torch.from_numpy(pad_mol_nmr(masked_mol_ids_list, 0)).long()
        atom_labels = torch.from_numpy(pad_mol_nmr(raw_mol_ids_list, 0)).long()
        masked_nmr = torch.from_numpy(pad_mol_nmr(masked_nmr, 0)).long()
        nmr_labels = torch.from_numpy(pad_mol_nmr(raw_nmr, 0)).long()
        masked_flag = torch.from_numpy(pad_mol_nmr(masked_flag, False))
        # masked_index =  torch.from_numpy(masked_index)
        return {'adjoin_matrix': adjoin_matrix_G,
                'masked_mol_ids': masked_mol_ids,
                'masked_nmr': masked_nmr,
                'nmr_labels': nmr_labels,
                "atom_labels": atom_labels,
                "masked_flag": masked_flag
                }



# class Pretrain_Masked_AtomNmr_Dataset_3D(Dataset):
#     def __init__(self,data = None, mol_fix_len=256):
#         super().__init__()
#         self.data = data
#         self.mol_fix_len = mol_fix_len
#         self.tokenizer = Tokenizer()
#
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __getitem__(self, index):
#         smi ,nmr= self.data[index][0],self.data[index][1]
#
#         # pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = [d.detach().numpy() for d in self.data[index][2:]]
#         pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = [d for d in self.data[index][2:]]
#         # smi, nmr  = self.data[index]
#         atom_symbol_list, adjoin_matrix = Smiles_to_adjoin(smi, explicit_hydrogens=True,size=self.mol_fix_len - 1)
#
#         # 预训练以及下游任务过滤NMR数值范围在[-800,1000],这里+801，变成[1,1801]
#         # <pad> <global> 对应的NMR值不参与运算，因此用任意值即可
#         # 而mask对应的NMR是需要参与计算的，使用0
#
#         nmr = nmr + 801
#
#         # 开头添加了<global>，(对应的nmr会被masked,因此设为任意值即可)
#         atom_symbol_list_interim = self.tokenizer.add_special_atom(atom_symbol_list)
#         nmr_interim = np.insert(nmr, 0,0)
#
#         # 截断
#         if len(atom_symbol_list) > self.mol_fix_len:
#             atom_symbol_list_interim = atom_symbol_list_interim[:self.mol_fix_len]
#             nmr_interim = nmr_interim[:self.mol_fix_len]
#
#         raw_nmr = copy.copy(nmr_interim)
#         raw_atom_symbol_list = copy.copy(atom_symbol_list_interim)
#         raw_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(raw_atom_symbol_list), np.int64)
#
#
#         masked_atom_symbol_list,masked_nmr,masked_flag = apply_mask(atom_symbol_list_interim, nmr_interim,self.tokenizer)
#
#
#         masked_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(masked_atom_symbol_list), np.int64)
#         masked_flag =  np.array(masked_flag,np.int64)
#
#         adjoin_matrix_G = np.ones([len(atom_symbol_list_interim),len(atom_symbol_list_interim)])
#         adjoin_matrix_G[1:, 1:] = adjoin_matrix
#
#         return  adjoin_matrix_G, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list,pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion
#
#     @staticmethod
#     def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
#         adjoin_matrix_G, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list,pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = tuple(zip(*batch))
#
#         batch_size = len(adjoin_matrix_G)
#         # 该batch最大的图的shape,其余的图需要对齐到该shape
#         shape = [batch_size] + [np.max([seq.shape for seq in adjoin_matrix_G])] * 2
#         dtype = adjoin_matrix_G[0].dtype
#
#         array = np.full(shape, 0, dtype=dtype)
#         for arr, matrix in zip(array, adjoin_matrix_G):
#             arrslice = tuple(slice(dim) for dim in matrix.shape)
#             arr[arrslice] = matrix
#         adjoin_matrix_G = torch.from_numpy(array)
#
#         # 将每个分子的nmr列表长度对齐（对齐到该batch中拥有最多原子的分子),
#         # <padding>对应的nmr不会参与运算，设为任意值即可（[0,1801],这里设为0
#         masked_mol_ids = torch.from_numpy(pad_mol_nmr(masked_mol_ids_list, 0)).long()
#         atom_labels = torch.from_numpy(pad_mol_nmr(raw_mol_ids_list, 0)).long()
#         masked_nmr = torch.from_numpy(pad_mol_nmr(masked_nmr, 0)).long()
#         nmr_labels = torch.from_numpy(pad_mol_nmr(raw_nmr, 0)).long()
#         masked_flag = torch.from_numpy(pad_mol_nmr(masked_flag, False))
#
#         return {'adjoin_matrix': adjoin_matrix_G,
#                 'masked_mol_ids': masked_mol_ids,
#                 'masked_nmr': masked_nmr,
#                 'nmr_labels': nmr_labels,
#                 "atom_labels": atom_labels,
#                 "masked_flag": masked_flag,
#                 "pos":pos,
#                 "dist_i":dist_i,
#                 "dist_j":dist_j,
#                 "dist":dist,
#                 "angle_i":angle_i,
#                 "angle_j":angle_j,
#                 "angle_k":angle_k,
#                 "angle":angle,
#                 "torsion_k":torsion_k,
#                 "torsion_i":torsion_i,
#                 "torsion_j":torsion_j,
#                 "torsion_t":torsion_t,
#                 "torsion":torsion,
#                 }
import bisect
import copy
from typing import Any, List, Dict
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import Tokenizer
import numpy as np

from utils import Smiles_to_adjoin, pad_mol_nmr, apply_mask


class Pretrain_Masked_AtomNmr_Dataset_3D(Dataset):
    def __init__(self,data = None, mol_fix_len=256):
        super().__init__()
        self.data = data
        self.mol_fix_len = mol_fix_len
        self.tokenizer = Tokenizer()


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):


        # pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = [d.detach().numpy() for d in self.data[index][2:]]

        smi, nmr,pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = self.data[index]
        nmr = np.array(nmr)
        atom_symbol_list, adjoin_matrix = Smiles_to_adjoin(smi, explicit_hydrogens=True,size=self.mol_fix_len - 1)

        nmr = nmr + 801

        atom_symbol_list_interim = self.tokenizer.add_special_atom(atom_symbol_list)
        nmr_interim = np.insert(nmr, 0,0)

        # 截断
        if len(atom_symbol_list) > self.mol_fix_len:
            atom_symbol_list_interim = atom_symbol_list_interim[:self.mol_fix_len]
            nmr_interim = nmr_interim[:self.mol_fix_len]

        raw_nmr = copy.copy(nmr_interim)
        raw_atom_symbol_list = copy.copy(atom_symbol_list_interim)
        raw_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(raw_atom_symbol_list), np.int64)


        masked_atom_symbol_list,masked_nmr,masked_flag = apply_mask(atom_symbol_list_interim, nmr_interim,self.tokenizer)


        masked_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(masked_atom_symbol_list), np.int64)
        masked_flag =  np.array(masked_flag,np.int64)

        adjoin_matrix_G = np.ones([len(atom_symbol_list_interim),len(atom_symbol_list_interim)])
        adjoin_matrix_G[1:, 1:] = adjoin_matrix


        return  adjoin_matrix_G, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list,pos,\
                dist_i.long(),dist_j.long(),dist,angle_i.long(),angle_j.long(),angle_k.long(),angle,\
                torsion_k.long(),torsion_i.long(),torsion_j.long(),torsion_t.long(),torsion

    @staticmethod
    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        adjoin_matrix_G, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list,pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,torsion = tuple(zip(*batch))
        batch_size = len(adjoin_matrix_G)
        # 该batch最大的图的shape,其余的图需要对齐到该shape
        padding_shape = np.max([seq.shape for seq in adjoin_matrix_G])
        shape = [batch_size] + [padding_shape] * 2
        dtype = adjoin_matrix_G[0].dtype

        array = np.full(shape, 0, dtype=dtype)
        for arr, matrix in zip(array, adjoin_matrix_G):
            arrslice = tuple(slice(dim) for dim in matrix.shape)
            arr[arrslice] = matrix
        adjoin_matrix_G = torch.from_numpy(array)

        # 将每个分子的nmr列表长度对齐（对齐到该batch中拥有最多原子的分子),
        # <padding>对应的nmr不会参与运算，设为任意值即可（[0,1801],这里设为0
        masked_mol_ids = torch.from_numpy(pad_mol_nmr(masked_mol_ids_list, 0)).long()
        atom_labels = torch.from_numpy(pad_mol_nmr(raw_mol_ids_list, 0)).long()
        masked_nmr = torch.from_numpy(pad_mol_nmr(masked_nmr, 0)).long()
        nmr_labels = torch.from_numpy(pad_mol_nmr(raw_nmr, 0)).long()
        masked_flag = torch.from_numpy(pad_mol_nmr(masked_flag, False))


        def index_helper(idx_tuple,padding_shape):
            # 第一个节点是global节点，需要跳过
            for i,idx in enumerate(idx_tuple):
                idx += 1
                idx += i * padding_shape


        for idx_tuple in [dist_i,dist_j,angle_i,angle_j,angle_k,torsion_k,torsion_i,torsion_j,torsion_t]:
            index_helper(idx_tuple,padding_shape)



        return {'adjoin_matrix': adjoin_matrix_G,
                'masked_mol_ids': masked_mol_ids,
                'masked_nmr': masked_nmr,
                'nmr_labels': nmr_labels,
                "atom_labels": atom_labels,
                "masked_flag": masked_flag,
                "pos":torch.concat(pos),
                "dist_i":torch.concat(dist_i) ,
                "dist_j":torch.concat(dist_j) ,
                "dist":torch.concat(dist),
                "angle_i":torch.concat(angle_i),
                "angle_j":torch.concat(angle_j) ,
                "angle_k":torch.concat(angle_k) ,
                "angle":torch.concat(angle),
                "torsion_k":torch.concat(torsion_k),
                "torsion_i":torch.concat(torsion_i),
                "torsion_j":torch.concat(torsion_j),
                "torsion_t":torch.concat(torsion_t),
                "torsion":torch.concat(torsion),

                }