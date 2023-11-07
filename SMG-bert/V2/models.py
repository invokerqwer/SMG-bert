

import logging
import sys
import os
sys.path.append("/home/zjh/remote/mrbert/V1/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from utils import Smiles_to_adjoin, pad_mol_nmr
import numpy as np
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict
import torch.nn as nn
import torch.nn.functional as F
import math

'''embedding层'''

class AtomEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim)

class NmrEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim)


'''注意力机制'''

# 0保留
class Attention(nn.Module):
    # q,k,v (bs,head,seq_len,emb)
    def forward(self, query, key, value, adjoin_matrix=None,mask=None):
        # (bs,head,seq_len_q,emb) @ (bs,head,seq_len_k,emb) => (bs,head,seq_len_q,seq_len_k)
        scaled_attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))

        # mask (bs,seq_len_q,seq_len_k),与scaled_attention_logits利用了广播机制，广播到head维度，即多个head使用的是同一份mask
        # mask由0,1组成；
        # 其中的1在乘以-1e9后会变成非常小的值，在softmax中会变成0，即qk分数为0;
        # 其中的0在乘以-1e9后会仍是0，在softmax中会有一定的值，即qk分数有一定值;
        # 即原始mask中的1表示不参与计算
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_scores = F.softmax(scaled_attention_logits, dim=-1)
        # 这个adjoin_matrix的生成方式可以考虑GCN + dist
        if adjoin_matrix is not None:
            attention_scores += adjoin_matrix
        # (bs,head,seq_len_q,seq_len_k)@ (bs,head,seq_len_v,emb)  => (bs,head,seq_len_q,emb)
        return torch.matmul(attention_scores, value), attention_scores

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embed_dim, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        assert embed_dim % head == 0
        self.head = head
        self.h_dim = embed_dim // head
        self.project_layer = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])
        self.attention = Attention()
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=p)
    def forward(self, query, key, value, mask=None,adjoin_matrix=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        batch_size = query.size(0)
        # (bsz,seq_len,emb_dim) => (bsz,seq_len,head_num,head_dim) => (bsz,head_num,seq_len,head_dim)
        query, key, value = [project(x).reshape(batch_size, -1, self.head, self.h_dim).transpose(1, 2)
                             for project, x in zip(self.project_layer, (query, key, value))]
        # (bs,head,seq_len_q,emb)
        x_value, attn = self.attention(query, key, value, mask=mask, adjoin_matrix=adjoin_matrix)
        # (bs,head,seq_len_q,emb) => (bs,seq_len_q,head,emb) => (bs,seq_len_q,head*emb)
        x_value = x_value.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.h_dim)
        out = self.output_linear(x_value)
        return out,attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, head, embed_dim, ffn,eps=1e-5, p=0.1, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.attention = MultiHeadedAttention(head=head, embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, ffn),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(ffn, embed_dim),
        )
    def forward(self, x, mask=None,adjoin_matrix=None):
        if self.norm_first:
            norm_x = self.norm1(x)
            x_value, attn = self.attention(norm_x, norm_x, norm_x, mask=mask, adjoin_matrix=adjoin_matrix)
            x = x + x_value
            x = x + self.ff_block(self.norm2(x))
        else:
            x_value, attn = self.attention(x, x, x, mask=mask, adjoin_matrix=adjoin_matrix)
            x = self.norm1(x + x_value)
            x = self.norm2(x + self.ff_block(x))
        return x

class Pretrain_MR_BERT_3D(nn.Module):
    def __init__(self, atom_vocab_size = 17, nmr_vocab_size = 1801, embed_dim = 128, ffn_dim = 256, head = 4, encoder_layers = 6, spec_endoder_layers=2, p=0.1,nmr=True,pos_info=True):
        super().__init__()
        self.pos_info = pos_info
        self.nmr = nmr
        self.atom_embedding = AtomEmbedding(atom_vocab_size,embed_dim)
        if self.nmr:
            self.nmr_embedding = NmrEmbedding(nmr_vocab_size,embed_dim)

        self.share_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(encoder_layers)])
        self.atom_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])
        self.atom_predict_block = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(ffn_dim, atom_vocab_size)
        )
        self.norm = nn.LayerNorm(embed_dim)

        if self.nmr:
            self.nmr_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])
            self.nmr_predict_block = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, 1)
            )
        if self.pos_info:
            self.pos_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])
            self.pos_predict_block = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, 3)
            )

    def forward(self, mol_ids_list, nmr_list, mask=None,adjoin_matrix=None):

        if self.nmr:
            x = self.atom_embedding(mol_ids_list) + self.nmr_embedding(nmr_list)
        else:
            x = self.atom_embedding(mol_ids_list)

        for transformer in self.share_transformer_blocks:
            x = transformer(x,mask=mask,adjoin_matrix=adjoin_matrix)

        x = self.norm(x)
        share = x
        nmr = pos =  None
        x_atom = x_nmr = x_pos = None

        for transformer in self.atom_transformer_blocks:
            x_atom = transformer(x,mask=mask,adjoin_matrix=adjoin_matrix)
        atom = self.atom_predict_block(x_atom)
        if self.nmr:
            for transformer in self.nmr_transformer_blocks:
                x_nmr = transformer(x,mask=mask,adjoin_matrix=adjoin_matrix)
            nmr = self.nmr_predict_block(x_nmr)
        if self.pos_info:
            for transformer in self.pos_transformer_blocks:
                x_pos = transformer(x,mask=mask,adjoin_matrix=adjoin_matrix)
            pos = self.pos_predict_block(x_pos)
        return nmr,atom,pos,x_atom,x_nmr,x_pos,share