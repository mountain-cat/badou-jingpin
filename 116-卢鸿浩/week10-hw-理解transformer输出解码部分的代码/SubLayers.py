''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):  # 2, 128, 64, 64
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # 128, 2*64
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # 128, 2*64
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # 128, 2*64
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)  # 2*64,128, 

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)  # 64**0.5=8

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 128


    def forward(self, q, k, v, mask=None):  # all is (batch_size, src_len, 128) (32,len,128)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head                      # -> 64,64,2
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  # -> 32, len1, len2, len2

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # (32,len1,128) -> (32,len1,2,64)  128是embedding的维度，这里分为了2个head
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # (32,len2,128) -> (32,len2,2,64)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # (32,len2,128) -> (32,len2,2,64)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) 
        # -> (32,2,len1,64) (batch_size, head_num, src_len1, 128/head_num)
        # -> (32,2,len2,64) (batch_size, head_num, src_len2, 128/head_num)
        # -> (32,2,len2,64) (batch_size, head_num, src_len2, 128/head_num)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # -> (32,2,len1,64) (32,2,len1,len2)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # ->(32,len1,head_num,64)->(32,len1,128) 合并多头
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
        # (batch_size, len1, d_model)        (32,len1,128) 
        # (batch_size, head_num, len1, len2) (32,2,len1,len2)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))  # 过一个线性层，再过一个relu，再过一个线性层
        x = self.dropout(x)
        x += residual                      # 加上原来的x

        x = self.layer_norm(x)             # 过归一化层

        return x
