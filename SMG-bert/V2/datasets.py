import random
import bisect
import copy
from typing import Any, List, Dict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import Tokenizer
import numpy as np

from utils import Smiles_to_adjoin, pad_mol_nmr, apply_mask


class Pretrain_Masked_AtomNmr_Dataset_2D(Dataset):
    def __init__(self,data = None, mol_fix_len=256):
        super().__init__()
        self.data = data
        self.mol_fix_len = mol_fix_len
        self.tokenizer = Tokenizer()
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        smi, nmr  = self.data[index]
        # mol_fix_len - 1是因为全局节点G
        atom_symbol_list, adjoin_matrix = Smiles_to_adjoin(smi, explicit_hydrogens=True,size=self.mol_fix_len - 1)

        # 预训练以及下游任务过滤NMR数值范围在[-800,1000],这里+801，变成[1,1801]
        # <pad> <global> 对应的NMR值不参与运算，因此用任意值即可
        # 而mask对应的NMR是需要参与计算的，使用0
        nmr = nmr + 801
        # 开头添加了<global>，(对应的nmr会被masked,因此设为任意值即可)
        atom_symbol_list_interim = self.tokenizer.add_special_atom(atom_symbol_list)
        nmr_interim = np.insert(nmr, 0, 0)

        # 截断
        if len(atom_symbol_list) > self.mol_fix_len:
            atom_symbol_list_interim = atom_symbol_list_interim[:self.mol_fix_len]
            nmr_interim = nmr_interim[:self.mol_fix_len]

        raw_nmr = nmr_interim
        raw_mol_ids_list = np.array(self.tokenizer.convert_atoms_to_ids(atom_symbol_list_interim), np.int64)
        # 这里注意深拷贝
        masked_mol_ids_list,masked_nmr,masked_flag = apply_mask(raw_mol_ids_list, nmr_interim)

        gcn_score = get_gcn_norm(adjoin_matrix,self_loop=True)
        mask = get_mask(adjoin_matrix)
        return  mask, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list,gcn_score
    @staticmethod
    def collate_fn(batch: List[Any]) -> Dict[str, torch.Tensor]:
        mask, masked_mol_ids_list, masked_nmr, raw_nmr,masked_flag, raw_mol_ids_list, gcn_score = tuple(zip(*batch))

        batch_size = len(mask)
        # 该batch最大的图的shape,其余的图需要对齐到该shape
        shape = [batch_size] + [np.max([seq.shape for seq in mask])] * 2
        array = np.full(shape, 1.0)
        for arr, matrix in zip(array, mask):
            arrslice = tuple(slice(dim) for dim in matrix.shape)
            arr[arrslice] = matrix
        mask = torch.from_numpy(array)
        array = np.full(shape, 0.0)
        for arr, matrix in zip(array, gcn_score):
            arrslice = tuple(slice(dim) for dim in matrix.shape)
            arr[arrslice] = matrix

        gcn_score = torch.from_numpy(array)
        # 将每个分子的nmr列表长度对齐（对齐到该batch中拥有最多原子的分子),
        # <padding>对应的nmr不会参与运算，设为任意值即可（[0,1801],这里设为0
        masked_mol_ids = torch.from_numpy(pad_mol_nmr(masked_mol_ids_list, 0)).long()
        atom_labels = torch.from_numpy(pad_mol_nmr(raw_mol_ids_list, 0)).long()
        masked_nmr = torch.from_numpy(pad_mol_nmr(masked_nmr, 0)).long()
        nmr_labels = torch.from_numpy(pad_mol_nmr(raw_nmr, 0)).long()
        masked_flag = torch.from_numpy(pad_mol_nmr(masked_flag, True))
        return {'adjoin_matrix': gcn_score,
                'masked_mol_ids': masked_mol_ids,
                'masked_nmr': masked_nmr,
                'nmr_labels': nmr_labels,
                "atom_labels": atom_labels,
                "masked_flag": masked_flag,
                "mask": mask,
                }
# 20% 概率会改变；其中80%是atom变成<mask>,nmr设为0，10%atom变成1~13，nmr变成[1,801]，剩下10%不做处理
def apply_mask(mol, nmr,mask_prob=0.15):

        masked_mol = mol.copy()
        masked_nmr = nmr.copy()
        lth = len(mol)
        # -1是防止索引越界 + 1是跳过G
        choices = np.random.permutation(lth - 1)[:max(int(lth * mask_prob), 1)] + 1
        masked_flag = np.zeros(lth)
        for i in choices:
            rand = np.random.rand()
            masked_flag[i] = 1
            if rand < 0.8:
                # 15是mask的编码
                masked_mol[i] = 15
                masked_nmr[i] = 0
            elif rand < 0.9:
                masked_mol[i] = random.randint(1, 14)
                masked_nmr[i] = random.randint(1, 801)
        masked_flag = np.array(masked_flag)
        return masked_mol, masked_nmr,masked_flag


from torch_sparse import SparseTensor
import torch_sparse

# mask 0 保留
# 给定邻接矩阵,有连接则生成0，这样在attention中不会被忽略
def get_mask(adjoin_matrix):
    adjoin_matrix = adjoin_matrix.copy()
    adjoin_matrix_G = np.ones([adjoin_matrix.shape[0]+1,adjoin_matrix.shape[1]+1])
    adjoin_matrix_G[1:, 1:] = adjoin_matrix
    return 1 - adjoin_matrix_G



# 用于的到attention score之后相加，因此这里的值越大表示分数越高
def get_gcn_norm(adjoin_matrix,self_loop=True):
    adjoin_matrix = adjoin_matrix.copy()
    adjoin_matrix_G = np.zeros([adjoin_matrix.shape[0]+1,adjoin_matrix.shape[1]+1])
    adjoin_matrix_G[1:, 1:] = norm_adj(adjoin_matrix,add_self_loops=self_loop)
    return adjoin_matrix_G


def norm_adj(adj_dense, add_self_loops=True):

    adj_dense = torch.tensor(adj_dense)
    adj_t = SparseTensor.from_dense(adj_dense)
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.)
    if add_self_loops:
        adj_t = torch_sparse.fill_diag(adj_t, 1.)
    deg = torch_sparse.sum(adj_t, dim=1)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t =torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t.to_dense().numpy()

# def get_dist_matrix(dist_i,dist_j,dist):
#     res  = torch.zeros([len(dist_i)]*2)
#     res[dist_i,dist_j] = dist
#
#     return res
