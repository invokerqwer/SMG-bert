import argparse
import os
import os
import os.path

import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from torch_sparse import SparseTensor
from tqdm import tqdm

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
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt() + (1e-10)
    plane1 = torch.cross(pos_ji, pos_jt)
    plane2 = torch.cross(pos_ji, pos_jk)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    torsion1 = torch.atan2(b, a) # -pi to pi
    torsion1 = torch.abs(torsion1)
    # torsion1[torsion1<=0]+=2*PI # 0 to 2pi
    return torsion1

def generate_dist_index(mol,valid=False,max_len=127):
    atom_num = min(max_len,mol.GetNumAtoms())
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
    atoms  = [atom for atom in mol.GetAtoms()]
    for index in range(min(max_len,mol.GetNumAtoms())):
        atom = atoms[index]
        set1 = set([bond.GetBeginAtomIdx() for bond in atom.GetBonds()])
        set2 = set([bond.GetEndAtomIdx() for bond in atom.GetBonds()])
        group = [i for i in list(set1 | set2) if i < max_len]
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
    if len(i_all) == 0:
        return None,None,None
    return i_all,j_all,k_all

def generate_torsion_index(mol,valid=False,max_len=127):

    atoms  = [atom for atom in mol.GetAtoms()]

    group_dict = {}
    bond_start = [i for i in [bond.GetBeginAtomIdx() for bond in mol.GetBonds()] if i < max_len]
    bond_end = [i for i in [bond.GetEndAtomIdx() for bond in mol.GetBonds()] if i< max_len]
    for index in range(min(max_len,mol.GetNumAtoms())):
        atom = atoms[index]
        set1 = set([bond.GetBeginAtomIdx() for bond in atom.GetBonds()])
        set2 = set([bond.GetEndAtomIdx() for bond in atom.GetBonds()])
        group = [i for i in list(set1 | set2) if i < max_len]
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
    if len(k_all) == 0:
        return  None,None,None,None
    return k_all,i_all,j_all,t_all

def generate_3d_info(data,dataset_path,max_len=127):
    count = 0
    smiles_list = []
    dist_list = []
    nmr_list = []
    angle_list = []
    torsion_list = []
    pos_list = []
    label_list = []
    for row in tqdm(iter(data),total=len(data)):
        smiles = row[0]
        label = row[1]
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
        except:
            print("can't get pos by rdkit")
            count += 1
            print(smiles)
            continue
        if pos.shape[0] > 127:
            print("smiles is longer than 127")
            print(smiles)
            continue
        dist_i,dist_j = generate_dist_index(mol,valid=True,max_len=max_len)
        angle_i,angle_j,angle_k = generate_angle_index(mol,valid=True,max_len=max_len)
        if angle_i is None:
            print("can't generate angle index")
            print(smiles)
            count += 1
            continue

        torsion_k,torsion_i,torsion_j,torsion_t = generate_torsion_index(mol,valid=True,max_len=max_len)
        if torsion_k is None:
            print("can't generate torsion index")
            print(smiles)
            count += 1
            continue

        dist = get_dist(pos,dist_i,dist_j)
        angle = get_angle(pos,angle_i,angle_j,angle_k)
        torsion = get_torsion(pos,torsion_k,torsion_i,torsion_j,torsion_t)

        if (len(dist)==0 or len(angle)==0 or len(torsion)==0):
            print("valid of length")
            print(smiles)
            count += 1
            continue
        try:
            spt = SparseTensor(row=dist_i.long(),col=dist_j.long(),value=dist)
            dist_matrix = spt.to_dense()
        except:
            print("cannot covert sparse adj to dense")
            print(smiles)
            count += 1
            continue

        smiles_list.append(smiles)
        dist_list.append(dist_matrix)
        pos_list.append(pos)
        angle_list.append([angle_i.byte(),angle_j.byte(),angle_k.byte(),angle])
        label_list.append(label)
        torsion_list.append([torsion_k.byte(),torsion_i.byte(),torsion_j.byte(),torsion_t.byte(),torsion])

    print(f"{count} is valid")
    base_dir = os.path.dirname(dataset_path)
    base_name = os.path.basename(dataset_path).split(".")[0]
    save_dir  = os.path.join(base_dir,base_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    smiles_save_path = os.path.join(save_dir,"smiles.pth")
    dist_save_path = os.path.join(save_dir,"dist.pth")
    nmr_save_path= os.path.join(save_dir,"nmr.pth")
    angle_save_path = os.path.join(save_dir,"angle.pth")
    torsion_save_path = os.path.join(save_dir,"torsion.pth")
    pos_save_path = os.path.join(save_dir,"pos.pth")
    label_save_path = os.path.join(save_dir,"label.pth")
    torch.save(smiles_list,smiles_save_path)
    torch.save(dist_list,dist_save_path)
    torch.save(nmr_list,nmr_save_path)
    torch.save(angle_list,angle_save_path)
    torch.save(torsion_list,torsion_save_path)
    torch.save(pos_list,pos_save_path)
    torch.save(label_list,label_save_path)


# python add_3d_info.py --dataset_path /home/zjh/mr/downstream/dataset/classify/chiral.pth
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int,default=127)
    parser.add_argument('--dataset_path', type=str, default="")
    args = parser.parse_args()
    data = torch.load(args.dataset_path)
    generate_3d_info(data, args.dataset_path, max_len=args.max_len)