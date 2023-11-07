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
    # q,k,v (bs,head,seq_len,emb)
    def forward(self, query, key, value, gnn_adj=None,dist_score=None,mask=None):
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

        if dist_score is not None:
            # print("jin dist")
            attention_scores = attention_scores+dist_score

        if gnn_adj is not None:
            # print("jin gnn")
            attention_scores = attention_scores+gnn_adj

        attention_scores = F.softmax(attention_scores, dim=-1)
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


        self.dist_block =  torch.nn.TransformerEncoderLayer(d_model=mol_fix_len,nhead=8,dim_feedforward=512)

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
    def __init__(self, atom_vocab_size = 18, embed_dim = 128, ffn_dim = 256, head = 4, encoder_layers = 6, spec_endoder_layers=2, p=0.1,pos_info=True):
        super().__init__()
        self.pos_info = pos_info
        self.atom_embedding = AtomEmbedding(atom_vocab_size,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.share_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(encoder_layers)])
        self.atom_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])

        self.atom_predict_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(ffn_dim, atom_vocab_size),
        ) for _ in range(3)])

        self.adj_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])

        self.adj_predict_block = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(ffn_dim, 2)
        )

        self.pos_transformer_blocks = nn.ModuleList([TransformerEncoderLayer(head=head, embed_dim=embed_dim,ffn=ffn_dim,p=p) for _ in range(spec_endoder_layers)])
        self.pos_predict_block = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(ffn_dim, 3)
        )

    def forward(self, mol_ids_list,dist_i,dist_j, mask=None,gnn_adj=None,dist_score=None):


        x = self.atom_embedding(mol_ids_list)

        for transformer in self.share_transformer_blocks:
            x,_ = transformer(x,mask=mask,gnn_adj=None,dist_score=None)
        x = self.norm(x)
        share = x
        x_atom = x_adj = x_pos = None
        # 1D 模块, GAT 的transformer , 只预测mask 原子
        for transformer in self.adj_transformer_blocks:
            x_atom,_ = transformer(x,mask=mask,gnn_adj=None,dist_score=None)
        atom_pred_1D = self.atom_predict_blocks[0](x_atom)
        # 2D 模块，引入了gnn_adj, 预测mask原子的同时也要重建邻接矩阵(二分类任务)
        # adj_pred 是3维坐标，可以用于计算距离
        for transformer in self.adj_transformer_blocks:
            x_adj,_ = transformer(x,mask=mask,gnn_adj=gnn_adj,dist_score=None)
        atom_pred_2D = self.atom_predict_blocks[1](x_adj)
        x_adj_tmp = x_adj.reshape(-1,x_adj.shape[-1])

        adj_pred = self.adj_predict_block((x_adj_tmp.index_select(0,dist_i) * x_adj_tmp.index_select(0,dist_j)))



        for transformer in self.pos_transformer_blocks:
            x_pos,dist_score = transformer(x,mask=None,gnn_adj=gnn_adj,dist_score=dist_score)
        pos_pred = self.pos_predict_block(x_pos)

        atom_pred_3D = self.atom_predict_blocks[2](x_adj)
        return atom_pred_1D,atom_pred_2D,adj_pred,pos_pred,atom_pred_3D,share,x_atom,x_adj,x_pos