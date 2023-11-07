import argparse
import os.path
from multiprocessing import Process

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
def generate_dist_index(mol,valid=False):
    atom_num = mol.GetNumAtoms()
    start = torch.arange(atom_num).repeat_interleave(atom_num)
    end = torch.arange(atom_num).repeat(atom_num)
    if valid:
        valid_index = start != end
        start = start[valid_index]
        end = end[valid_index]
    return start,end
def generate_angle_index(mol,valid=False):
    i_all = []
    j_all = []
    k_all = []
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
    return i_all,j_all,k_all

def generate_torsion_index(mol,valid=False):
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
    return k_all,i_all,j_all,t_all


def process(data,dataset_dir,file_name):
    new_data = []
    id = 0
    error_id = []
    for row in tqdm(iter(data),total=len(data)):


        smiles = row[0]
        nmr = row[1]
        try:

            mol = AllChem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            pos=[]
            edge_index=[]
            num_nodes=len(mol.GetAtoms())
            for i in range(num_nodes):
                for j in range(i):
                    edge_index.append([i,j])
                x,y,z =mol.GetConformer().GetAtomPosition(i)
                pos.append([x,y,z])
            pos = torch.tensor(pos)
            dist_i,dist_j = generate_dist_index(mol,valid=True)
            angle_i,angle_j,angle_k = generate_angle_index(mol,valid=True)
            torsion_k,torsion_i,torsion_j,torsion_t = generate_torsion_index(mol,valid=True)
            dist = get_dist(pos,dist_i,dist_j)
            angle = get_angle(pos,angle_i,angle_j,angle_k)
            torsion = get_torsion(pos,torsion_k,torsion_i,torsion_j,torsion_t)
            new_data.append([smiles,nmr,pos,dist_i,dist_j,dist,angle_i,angle_j,angle_k,angle,torsion_k,torsion_i,torsion_j,torsion_t,
                             torsion])
        except:
            error_id.append(id)
            continue
        id += 1

    torch.save(new_data,os.path.join(dataset_dir+"_3D",file_name))
dataset_dir = "/home/zjh/mr/downstream/dataset/classify"


if __name__=='__main__':
    p_l=[]
    for file_name in os.listdir(dataset_dir):
        data = torch.load(os.path.join(dataset_dir, file_name))
        p=Process(target=process,args=(data,dataset_dir,file_name))
        p_l.append(p)
        p.start()
    for p in p_l:
        p.join()
