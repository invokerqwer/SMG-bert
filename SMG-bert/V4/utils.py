import copy
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch_sparse
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
from rdkit import RDLogger
from torch_sparse import SparseTensor

RDLogger.DisableLog('rdApp.*')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

"""
padding,对齐到该batch的最长序列
"""
def pad_mol_nmr(batch_mol, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(batch_mol)
    # (batch_size,max_mol_ids_length)
    shape = [batch_size] + np.max([mol.shape for mol in batch_mol], 0).tolist()

    if dtype is None:
        dtype = batch_mol[0].dtype
    array = np.full(shape, constant_value, dtype=dtype)
    for arr, mol in zip(array, batch_mol):  # 补全
        arrslice = tuple(slice(dim) for dim in mol.shape)
        arr[arrslice] = mol
    return array
# 20% 概率会改变；其中80%是atom变成<mask>,nmr设为0，10%atom变成1~13，nmr变成[1,801]，剩下10%不做处理
# padding之前使用
def apply_atom_mask(atom_ids,raw_lth,mask_prob=0.15):

        # -1是防止索引越界 + 1是跳过G
        choices = np.random.permutation(raw_lth - 1)[:max(int(raw_lth * mask_prob), 1)] + 1
        masked_flag = [0] * len(atom_ids)
        for i in choices:
            rand = np.random.rand()
            masked_flag[i] = 1
            if rand < 0.8:
                # 15是mask的编码
                atom_ids[i] = 15
            elif rand < 0.9:
                atom_ids[i] = random.randint(1, 14)
        return atom_ids,masked_flag

def gcn_norm_adj(adj_dense, add_self_loop=True):
    adj_t = SparseTensor.from_dense(adj_dense)
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.)
    if add_self_loop:
        adj_t = torch_sparse.fill_diag(adj_t, 1.)
    deg = torch_sparse.sum(adj_t, dim=1)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t =torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t.to_dense()


def Smiles_to_adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False,size=127):
    mol = Chem.MolFromSmiles(smiles)
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)
    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    atom_symbol_list = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # ——————————————————————————
    num_atoms = mol.GetNumAtoms()
    adjoin_matrix = torch.eye(num_atoms)
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0

    return atom_symbol_list[:size], adjoin_matrix[:size,:size]





def calculate_recovery(atom_pred,atom_labels,masked_flag):
    atom_pred = atom_pred[masked_flag==1]
    atom_pred = torch.softmax(atom_pred,-1)
    atom_labels = atom_labels[masked_flag==1]
    atom_pred_labels = torch.argmax(atom_pred, -1)
    return torch.sum(atom_labels ==  atom_pred_labels) / len(atom_labels)


def get_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path:
        file_handler = logging.FileHandler(log_path,mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger


def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 20% 概率会改变；其中80%是atom变成<mask>,nmr设为0，10%atom变成1~13，nmr变成[1,801]，剩下10%不做处理
def apply_mask(mol, nmr, tokenizer,mask_prob=0.8):
        masked_mol = copy.copy(mol)
        masked_nmr = copy.copy(nmr)
        masked_flag = []
        for i, atom in enumerate(masked_mol):

            if atom in (tokenizer.start_token):
                masked_flag.append(False)
                continue
            prob = random.random()
            if prob < 1- mask_prob:
                masked_flag.append(True)
                # labels[i] = self.tokenizer.convert_atom_to_id(atom)
                prob /= (1- mask_prob)
                if prob < 0.8:  # 80% random change to mask token
                    atom = tokenizer.mask_token
                    masked_nmr[i] = 0
                elif prob < 0.9:  # 10% chance to change to random token
                    atom = tokenizer.convert_id_to_atom(random.randint(1, 14))
                    masked_nmr[i] = random.randint(1, 801)
                else:  # 10% chance to keep current token
                    # masked_nmr[i] = random.randint(100, 1800)
                    pass
                masked_mol[i] = atom
            else:
                masked_flag.append(False)
        masked_flag = np.array(masked_flag)
        return masked_mol, masked_nmr,masked_flag



def get_dist(pos,i,j):
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    return dist

def get_angle(pos,i,j,k):
        # Calculate angles. 0 to pi
    pos_ji = pos[i] - pos[j]
    pos_jk = pos[k] - pos[j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle

def get_torsion(pos,k,i,j,t):
    pos_jt = pos[j] - pos[t]
    pos_ji = pos[j] - pos[i]
    pos_jk = pos[j] - pos[k]
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt() + (1e-10)
    plane1 = torch.cross(pos_ji, pos_jt)
    plane2 = torch.cross(pos_ji, pos_jk)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    torsion1 = torch.atan2(b, a) # -pi to pi
    torsion1 = torch.abs(torsion1)
    return torsion1