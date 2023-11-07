import argparse
import os.path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from torch_sparse import SparseTensor
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import torch


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
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(pos_ji, pos_jt)
    plane2 = torch.cross(pos_ji, pos_jk)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    torsion1 = torch.atan2(b, a) # -pi to pi
    torsion1 = torch.abs(torsion1)
    # torsion1[torsion1<=0]+=2*PI # 0 to 2pi
    return torsion1


def generate_dist_index(mol,valid=False,max_len=127):
    atom_num = mol.GetNumAtoms()
    atom_num = min(atom_num,max_len)
    start = torch.arange(atom_num).repeat_interleave(atom_num)
    end = torch.arange(atom_num).repeat(atom_num)
    if valid:
        valid_index = start != end
        start = start[valid_index]
        end = end[valid_index]
    return start,end





def generate_angle_index(mol,valid=False,max_len=127):
    i_all = []
    j_all = []
    k_all = []
    try:
        atoms  = [atom for atom in mol.GetAtoms()]
        for index in range(mol.GetNumAtoms()):
            atom = atoms[index]
            set1 = set([bond.GetBeginAtomIdx() for bond in atom.GetBonds()])
            set2 = set([bond.GetEndAtomIdx() for bond in atom.GetBonds()])
            group = list(set1 | set2)
            lth = len(group)
            if len(group) <= 2:
                continue
            i = torch.tensor(group).repeat_interleave(lth)
            j = torch.ones(lth**2,dtype=torch.int64) * index
            k = torch.tensor(group).repeat(lth)
            i_all.append(i)
            j_all.append(j)
            k_all.append(k)
        i_all = torch.cat(i_all)
        j_all = torch.cat(j_all)
        k_all = torch.cat(k_all)
        if valid:
            valid_index = (i_all != k_all) & (i_all != j_all) & (k_all != j_all)
            i_all = i_all[valid_index]
            j_all = j_all[valid_index]
            k_all = k_all[valid_index]
        idxs = []
        for idx in range(len(i_all)):
            i = i_all[idx]
            j = j_all[idx]
            k = k_all[idx]
            if i > max_len or j > max_len or k > max_len:
                continue
            else:
                idxs.append(idx)
    except:
        return [],[],[]
    return i_all[idxs],j_all[idxs],k_all[idxs]


def generate_torsion_index(mol,valid=False,max_len=127):
    atoms  = [atom for atom in mol.GetAtoms()]
    group_dict = {}
    bond_start = [bond.GetBeginAtomIdx() for bond in mol.GetBonds()]
    bond_end = [bond.GetEndAtomIdx() for bond in mol.GetBonds()]
    for index in range(mol.GetNumAtoms()):
        atom = atoms[index]
        set1 = set([bond.GetBeginAtomIdx() for bond in atom.GetBonds()])
        set2 = set([bond.GetEndAtomIdx() for bond in atom.GetBonds()])
        group = list(set1 | set2)
        group_dict[index] = group
    i_all = []
    j_all = []
    k_all =[]
    t_all = []
    for atom_i,atom_j in zip(bond_start,bond_end):
        i_group = group_dict[atom_i]
        j_group = group_dict[atom_j]
        k = torch.tensor(i_group).repeat_interleave(len(j_group))
        t = torch.tensor(j_group).repeat(len(i_group))
        i = torch.ones(len(j_group)*len(i_group),dtype=torch.int64) * atom_i
        j = torch.ones(len(j_group)*len(i_group),dtype=torch.int64) * atom_j
        i_all.append(i)
        j_all.append(j)
        k_all.append(k)
        t_all.append(t)
    i_all = torch.cat(i_all)
    j_all = torch.cat(j_all)
    k_all = torch.cat(k_all)
    t_all = torch.cat(t_all)
    if valid:
        valid_index = (k_all != t_all) & (k_all != i_all) & (k_all != j_all) & (t_all != i_all) & (t_all != j_all)
        i_all = i_all[valid_index]
        j_all = j_all[valid_index]
        k_all = k_all[valid_index]
        t_all = t_all[valid_index]
    idxs = []
    for idx in range(len(i_all)):
        i = i_all[idx]
        j = j_all[idx]
        k = k_all[idx]
        t = t_all[idx]
        if i > max_len or j > max_len or k > max_len or t > max_len:
            continue
        else:
            idxs.append(idx)
    return k_all[idxs],i_all[idxs],j_all[idxs],t_all[idxs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--smiles_col_name', type=str)
    parser.add_argument("--label_col_name", type=str)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--save_dir", type=str)

    args = parser.parse_args()

    data = pd.read_csv(args.file_path)
    label_col_name = args.label_col_name
    smiles_col_name = args.smiles_col_name
    save_dir =  args.save_dir
    labels = data[label_col_name]
    smiles_s = data[smiles_col_name]

    smiles_list = []
    dist_list = []
    nmr_list = []
    angle_list = []
    torsion_list = []
    pos_list = []
    count = 0
    label_list = []

    for smiles, label in tqdm(iter(zip(smiles_s, labels)), total=len(smiles_s)):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"invalid smiles {smiles}")
            continue

        mol = AllChem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            pos = []
            edge_index = []
            num_nodes = len(mol.GetAtoms())
            for i in range(num_nodes):
                for j in range(i):
                    edge_index.append([i, j])
                x, y, z = mol.GetConformer().GetAtomPosition(i)
                pos.append([x, y, z])
        except:
            print(f"ValueError: Bad Conformer Id {smiles}")
            continue


        pos = torch.tensor(pos)
        dist_i, dist_j = generate_dist_index(mol, valid=True)
        angle_i, angle_j, angle_k = generate_angle_index(mol, valid=True)
        torsion_k, torsion_i, torsion_j, torsion_t = generate_torsion_index(mol, valid=True)
        dist = get_dist(pos, dist_i, dist_j)
        angle = get_angle(pos, angle_i, angle_j, angle_k)
        torsion = get_torsion(pos, torsion_k, torsion_i, torsion_j, torsion_t)

        if (len(dist) == 0 or len(angle) == 0 or len(torsion) == 0):
            print("invalid dist/angle/torsion")
            print(smiles)
            continue

        try:
            spt = SparseTensor(row=dist_i.long(), col=dist_j.long(), value=dist)
            dist_matrix = spt.to_dense()
        except:
            print("valid cannot covert adj dense")
            print(smiles)
            continue

        smiles_list.append(smiles)
        dist_list.append(dist_matrix)
        pos_list.append(pos)
        # nmr_list.append(nmr)
        angle_list.append([angle_i.byte(), angle_j.byte(), angle_k.byte(), angle])
        label_list.append(label)

        torsion_list.append([torsion_k.byte(), torsion_i.byte(), torsion_j.byte(), torsion_t.byte(), torsion])
        count += 1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    smiles_save_path = os.path.join(save_dir, "smiles.pth")
    dist_save_path = os.path.join(save_dir, "dist.pth")
    nmr_save_path = os.path.join(save_dir, "nmr.pth")
    angle_save_path = os.path.join(save_dir, "angle.pth")
    torsion_save_path = os.path.join(save_dir, "torsion.pth")
    pos_save_path = os.path.join(save_dir, "pos.pth")
    label_save_path = os.path.join(save_dir, "label.pth")
    torch.save(smiles_list, smiles_save_path)
    torch.save(dist_list, dist_save_path)
    torch.save(nmr_list, nmr_save_path)
    torch.save(angle_list, angle_save_path)
    torch.save(torsion_list, torsion_save_path)
    torch.save(pos_list, pos_save_path)
    torch.save(label_list, label_save_path)
