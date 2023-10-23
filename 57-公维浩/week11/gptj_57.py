import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
'''
手动实现一个gpt-j形式的attention前向计算过程，参考第六周transformer的手动实现
'''

class DiyAttention(nn.Module):
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        super(DiyTransformer, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 64
        self.hidden_size = 768
        self.num_layers = 1
        self.load_weights(state_dict)
        self.q = nn.Linear(self.hidden_size, self.hidden_size)
        self.k = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.ffn_intermediate = nn.Linear(self.hidden_size, 3072)
        self.ffn_output = nn.Linear(3072, self.hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.pooler_output_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.Softmax()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def load_weights(self, state_dict):
        # embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()  # vocab_size * hidden_size
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()  # max_len * hidden_size
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()

    # bert embedding，使用3层叠加，在经过一个embedding层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        embedding = torch.Tensor(embedding)  # pytorch形式输入
        return embedding

    # embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    # 多头机制
    def transpose_for_scores(self, x):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, self.num_attention_heads, self.attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    def forword(self, x):
        hidden_size = self.hidden_size
        embedding = self.embedding_forward(x)  # batch_size, sen_len, embedding_size
        # 加和后有一个归一化层
        x = self.embedding_layer_norm(embedding)  # shpae: [max_len, hidden_size]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        qk = torch.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(self.attention_head_size)
        qk = self.softmax(qk)
        qkv = torch.matmul(qk, v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        attention = self.attention_linear(qkv)
        ffn_intermediate = self.ffn_intermediate(x)
        ffn_intermediate = self.gelu(ffn_intermediate)
        ffn_output = self.ffn_output(ffn_intermediate)
        sequence_output = self.layer_norm(x + ffn_output + attention)  # gpt-j结构
        pooler_output = self.tanh(self.pooler_output_layer(sequence_output[0]))
        return sequence_output, pooler_output


# 输入数据
x = np.array([2450, 15486, 15167, 2110])  # 通过vocab对应输入：深度学习
bert = BertModel.from_pretrained(r"E:\Code\model\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
dt = DiyAttention(state_dict)
sequence_output, pooler_output = dt.forword(x)
# print(diy_sequence, diy_pooler)
print(sequence_output.shape, pooler_output.shape)
