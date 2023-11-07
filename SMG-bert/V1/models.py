import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

'''embedding层'''

class AtomEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim)



class NmrEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__(vocab_size, embed_dim)


'''注意力机制'''

class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        # q_scaled = query / math.sqrt(query.size(-1))
        # torch.baddbmm(mask, q_scaled, key.transpose(-2, -1))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:  # 即 mask = adjoin_matrix
            scores = (scores * mask).float()
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_scores = F.softmax(scores, dim=-1)
        if mask is not None:
            attention_scores = torch.where(torch.isnan(attention_scores), torch.full_like(attention_scores, 0),
                                           attention_scores)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores


# ————————————————————————————————————————————————————————————————————————————————
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

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # (bsz,seq_len,emb_dim) => (bsz,seq_len,head_num,head_dim)
        query, key, value = [project(x).view(batch_size, -1, self.head, self.h_dim).transpose(1, 2)
                             for project, x in zip(self.project_layer, (query, key, value))]

        x_value, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x_value = x_value.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.h_dim)
        out = self.output_linear(x_value)
        return out


#######################################################################################


class TransformerEncoderLayer(nn.Module):
    def __init__(self, head, embed_dim, eps=1e-5, p=0.1, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.attention = MultiHeadedAttention(head=head, embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, mask):
        if self.norm_first:
            norm_x = self.norm1(x)
            x = x + self.attention(norm_x, norm_x, norm_x, mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.attention(x, x, x, mask))
            x = self.norm2(x + self.ff_block(x))
        return x


'''Embedding层 + TransformerBlock = Bert'''


class MR_BERT(nn.Module):
    def __init__(self, atom_vocab_size=17, nmr_vocab_size=1801, embed_dim=128, ffn_dim=256, head=4, encoder_layers=6,
                 p=0.1,nmr=True):
        super().__init__()
        self.nmr = nmr
        self.atom_embedding = AtomEmbedding(atom_vocab_size,embed_dim)
        if self.nmr:
            self.nmr_embedding = NmrEmbedding(nmr_vocab_size,embed_dim)

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(head=head, embed_dim=embed_dim, p=p) for _ in range(encoder_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.downstream_predict_block = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            nn.Linear(ffn_dim, ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            nn.Linear(ffn_dim, 1)
        )

    def forward(self, mol_ids_list, nmr_list, mask):
        if self.nmr:
            x = self.atom_embedding(mol_ids_list) + self.nmr_embedding(nmr_list)
        else:
            x = self.atom_embedding(mol_ids_list)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        x = self.norm(x)
        x = x[:, 0, :]
        pred = self.downstream_predict_block(x)
        return pred
class Pretrain_MR_BERT(nn.Module):
    def __init__(self, atom_vocab_size=17, nmr_vocab_size=1801, embed_dim=128, ffn_dim=256, head=4, encoder_layers=6,
                 p=0.1,nmr=True):
        super().__init__()
        self.nmr = nmr
        self.atom_embedding = AtomEmbedding(atom_vocab_size,embed_dim)
        if self.nmr:
            self.nmr_embedding = NmrEmbedding(nmr_vocab_size,embed_dim)

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(head=head, embed_dim=embed_dim, p=p) for _ in range(encoder_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.predict_block = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            nn.Linear(ffn_dim, ffn_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            nn.Linear(ffn_dim, atom_vocab_size)
        )

    def forward(self, mol_ids_list, nmr_list, mask):
        if self.nmr:
            x = self.atom_embedding(mol_ids_list) + self.nmr_embedding(nmr_list)
        else:
            x = self.atom_embedding(mol_ids_list)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        x = self.norm(x)
        pred = self.predict_block(x)
        return pred

'''Embedding层 + TransformerBlock = Bert'''

class Pretrain_MR_BERT_3D(nn.Module):
    def __init__(self, atom_vocab_size = 17, nmr_vocab_size = 1801, embed_dim = 128, ffn_dim = 256, head = 4, encoder_layers = 6, spec_endoder_layers=2, p=0.1,
                 nmr=True,pos_info=True):
        super().__init__()

        self.pos_info = pos_info

        self.nmr = nmr
        self.atom_embedding = AtomEmbedding(atom_vocab_size,embed_dim)
        if self.nmr:
            self.nmr_embedding = NmrEmbedding(nmr_vocab_size,embed_dim)
        self.transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,p=p) for _ in range(encoder_layers)])


        self.atom_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,p=p) for _ in range(spec_endoder_layers)])
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
            self.nmr_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,p=p) for _ in range(spec_endoder_layers)])
            self.nmr_predict_block = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, 1)
            )
        if self.pos_info:
            self.pos_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,p=p) for _ in range(spec_endoder_layers)])
            self.pos_predict_block = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, 3)
            )

    def forward(self, mol_ids_list, nmr_list, mask,):
        if self.nmr:
            x = self.atom_embedding(mol_ids_list) + self.nmr_embedding(nmr_list)
        else:
            x = self.atom_embedding(mol_ids_list)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        x = self.norm(x)

        share = x
        nmr = pos =  None
        x_nmr = x_pos = None
        x_atom = x
        for transformer in self.atom_transformer_blocks:
            x_atom = transformer(x_atom, mask)
        atom = self.atom_predict_block(x_atom)



        if self.nmr:
            x_nmr = x
            for transformer in self.nmr_transformer_blocks:
                x_nmr = transformer(x_nmr, mask)
            nmr = self.nmr_predict_block(x_nmr)

        if self.pos_info:
            x_pos = x
            for transformer in self.pos_transformer_blocks:
                x_pos = transformer(x_pos, mask)
            pos = self.pos_predict_block(x_pos)

        return nmr,atom,pos,x_atom,x_nmr,x_pos,share