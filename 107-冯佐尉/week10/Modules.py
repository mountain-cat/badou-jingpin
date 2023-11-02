import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):  # (32,2,len1,64) (32,2,len2,64) (32,2,len2,64)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # (32,2,len1,64) (32,2,64,len2)->(32,2,len1,len2)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # mask中为0的位置，写成一个很小的数

        attn = self.dropout(F.softmax(attn, dim=-1))  # 这个就是attention
        output = torch.matmul(attn, v)                                # (32,2,len1,len2) (32,2,len2,64)->(32,2,len1,64)

        return output, attn  # (32,2,len1,64) (32,2,len1,len2)
