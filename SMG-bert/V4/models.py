import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, query, key, value, mask=None, gnn_adj=None,dist_score=None):




        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))


        if mask is not None:  # 即 mask = adjoin_matrix
            scores = (scores * mask).float()
            scores = scores.masked_fill(mask == 0, float("-inf"))


        attention_scores = F.softmax(scores, dim=-1)
        if dist_score is not None:

            attention_scores = attention_scores+dist_score

        if gnn_adj is not None:

            attention_scores = attention_scores+gnn_adj
        # attention_scores = F.softmax(attention_scores, dim=-1)
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

    def forward(self, query, key, value, mask=None,gnn_adj=None,dist_score=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        if gnn_adj is not None:
            gnn_adj = gnn_adj.unsqueeze(1)
        if dist_score is not None:
            dist_score = dist_score.unsqueeze(1)

        batch_size = query.size(0)
        # (bsz,seq_len,emb_dim) => (bsz,seq_len,head_num,head_dim) => (bsz,head_num,seq_len,head_dim)
        query, key, value = [project(x).reshape(batch_size, -1, self.head, self.h_dim).transpose(1, 2)
                             for project, x in zip(self.project_layer, (query, key, value))]
        # (bs,head,seq_len_q,emb)
        x_value, attn = self.attention(query, key, value, mask=mask, gnn_adj=gnn_adj,dist_score=dist_score)
        # (bs,head,seq_len_q,emb) => (bs,seq_len_q,head,emb) => (bs,seq_len_q,head*emb)
        x_value = x_value.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.h_dim)
        out = self.output_linear(x_value)
        return out,attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, head, embed_dim, ffn,eps=1e-5, p=0.1, norm_first=True,mol_fix_len=128):
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
        self.dist_block = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=mol_fix_len,nhead=8,dim_feedforward=512),num_layers=3)

    def forward(self, x,mask=None,gnn_adj=None,dist_score=None):
        dist_score_raw = None
        if dist_score is not None:
            dist_score_raw =  self.dist_block(dist_score)
            dist_score = dist_score_raw[:,:x.shape[1],:x.shape[1]]
        if self.norm_first:
            norm_x = self.norm1(x)
            x_value, attn = self.attention(norm_x, norm_x, norm_x, mask=mask, gnn_adj=gnn_adj,dist_score=dist_score)
            x = x + x_value
            x = x + self.ff_block(self.norm2(x))
        else:
            x_value, attn = self.attention(x, x, x, mask=mask, gnn_adj=gnn_adj,dist_score=dist_score)
            x = self.norm1(x + x_value)
            x = self.norm2(x + self.ff_block(x))
        return x,dist_score_raw

class Pretrain_MR_BERT(nn.Module):
    def __init__(self, atom_vocab_size = 18, embed_dim = 128, ffn_dim = 256, head = 4, encoder_layers = 6, spec_endoder_layers=2, p=0.1,use_3D=True,use_1D=True,use_adj_ssl=True):
        super().__init__()
        self.use_1D = use_1D
        self.use_3D = use_3D
        self.use_adj_ssl = use_adj_ssl

        self.atom_embedding = AtomEmbedding(atom_vocab_size,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.share_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(encoder_layers)])

        self.transformer_blocks_2D = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim, ffn=ffn_dim, p=p) for _ in range(spec_endoder_layers)])
        self.atom_predict_block_2D = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(ffn_dim, atom_vocab_size),
        )

        if use_adj_ssl:
            self.adj_predict_block = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, 2)
            )

        if self.use_1D:
            self.transformer_blocks_1D = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])
            self.atom_predict_block_1D = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, atom_vocab_size),
            )


        if self.use_3D:
            self.transformer_blocks_3D = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])
            self.pos_predict_block = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, 3)
            )
            self.atom_predict_block_3D = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, ffn_dim),
                nn.Dropout(p=p),
                nn.ReLU(),
                nn.Linear(ffn_dim, atom_vocab_size),
            )

    def forward(self, mol_ids_list,dist_i,dist_j, mask=None,gnn_adj=None,dist_score=None):

        x = self.atom_embedding(mol_ids_list)

        # 这里采用GAT
        for transformer in self.share_transformer_blocks:
            x,_ = transformer(x,mask=mask,gnn_adj=None,dist_score=None)
        x = self.norm(x)
        share = x

        x_1D = x_2D = x_3D = None

        # 2D 模块 同mg-bert
        for transformer in self.transformer_blocks_2D:
            x_2D,_ = transformer(x,mask=mask,gnn_adj=gnn_adj,dist_score=None)
        x_2D = self.norm2(x_2D)
        atom_pred_2D = self.atom_predict_block_2D(x_2D)

        adj_pred = None
        if self.use_adj_ssl:
            x_adj_tmp = x_2D.reshape(-1, x_2D.shape[-1])
            adj_pred = self.adj_predict_block((x_adj_tmp.index_select(0, dist_i) * x_adj_tmp.index_select(0, dist_j)))

        atom_pred_1D = None
        if self.use_1D:
            # 1D 模块使用标准的attention
            for transformer in self.transformer_blocks_1D:
                x_1D,_ = transformer(x,mask=None,gnn_adj=None,dist_score=None)
            x_1D = self.norm1(x_1D)
            atom_pred_1D = self.atom_predict_block_1D(x_1D)

        pos_pred_3D =None
        atom_pred_3D = None
        if self.use_3D:
            # 3D GAT基础上融入dist_score,加不加gnn_adj看情况
            for transformer in self.transformer_blocks_3D:
                x_3D, dist_score = transformer(x, mask=mask, gnn_adj=gnn_adj, dist_score=dist_score)
            x_3D = self.norm3(x_3D)
            pos_pred_3D= self.pos_predict_block(x_3D)
            atom_pred_3D = self.atom_predict_block_3D(x_3D)

        return atom_pred_2D, adj_pred, atom_pred_1D,atom_pred_3D, pos_pred_3D, x_1D,x_2D,x_3D, share