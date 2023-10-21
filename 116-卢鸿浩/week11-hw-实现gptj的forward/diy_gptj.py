import torch
import math
import numpy as np
from transformers import BertModel

###

#通过手动矩阵运算实现GPTJ结构
#相比bert的主要区别是，这里计算attention和forward之后，和原始输入相加，送入归一化层
#手动敲了一下整个流程，理解更加透彻了

###


bert = BertModel.from_pretrained(r"E:\works\codes\python\bert-bass-chinese\bert-base-chinese", return_dict = False) #?
state_dict = bert.state_dict() #返回一个字典，其中包含模块所有state的引用
bert.eval()
x = np.array([2450, 15486, 15167, 2110]) #
torch_x = torch.LongTensor([x])

seq_output, pooler_output = bert(torch_x)
print(seq_output.shape, pooler_output.shape)
print(seq_output, pooler_output) 

print(state_dict.keys())  # 查看所有权值矩阵名称

# softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = -1, keepdims=True)

def gelu(x):
    return 0.5 * x * \
        (1 + np.tanh( math.sqrt(2/math.pi)) * (x + 0.044715*np.power(x,3) ) )

class DiyGptj:
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1
        self.load_weights(state_dict)
        self.output_layer_norm_w = np.random.rand(self.hidden_size)
        self.output_layer_norm_b = np.random.rand(self.hidden_size)
        
    # copyed
    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
    
    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x) # ->shape: [max_len, hidden_len]
        
        # position embedding的输入是[0, 1, 2, 3]
        pe = self.get_embedding(self.word_embeddings, np.array(list(range(len(x))))) # ->shape: 同上
        print("position embedding:")
        print(pe)
        print(self.embeddings_layer_norm_weight.shape)
        
        # token embedding单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.word_embeddings, np.array([0]*len(x))) # ->shape: 同上
        embedding = we + pe + te
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embedding

    def get_embedding(self, embedding_matrix, x):
        # 找到输入x中每一个值在embedding_matrix中的对应的向量
        # shape: max_len -> [male_len, hidden_size]
        return np.array([embedding_matrix[index] for index in x])

    # 执行全部层的transformer
    def all_gptj_layer_forward(self,x):
        for i in range(self.num_layers):
            x = self.single_gptj_layer_forward(x, i)
        return x

    # 执行单层的transformer
    def single_gptj_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights

        attention_output = self.self_attention(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bias,
                                               self.num_attention_heads,
                                               self.hidden_size
                                               )

        feed_forward_x = self.feed_forward(x,    
                        intermediate_weight,
                        intermediate_bias,
                        output_weight,
                        output_bias,)
        
        # GPTJ结构
        x = self.layer_norm(x + attention_output + feed_forward_x, self.output_layer_norm_w, self.output_layer_norm_b)
        return x

    # self_attention计算
    def self_attention(self,
                    x,
                    q_w, q_b,
                    k_w, k_b,
                    v_w, v_b,
                    attention_output_weight, attention_output_bias,
                    num_attention_heads,
                    hidden_size
                    ):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w shape = hidden_size*hidden_size
        # q_b, q_b, v_b shape = hidden_size
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]
        k = np.dot(x, k_w.T) + k_b
        v = np.dot(x, v_w.T) + v_b
        attention_head_size = int(hidden_size / num_attention_heads)
        #__split shape:[num_attention_heads, max_len, attention_head_size]
        #              max_len = num_attention_heads * attention_head_size
        q_split = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        k_split = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        v_split = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # print(k_split.shape())
        qk = np.matmul(q_split, k_split.swapaxes(1,2))  #-> shpae: [num_heads, max_len, max_len]
        qk /= np.sqrt(attention_head_size) #-> shape: still
        qk = softmax(qk)                   #-> shape: still

        qkv = np.matmul(qk, v_split)              #-> shape: [num_heads, max_len, attention_head_size]
        qkv = qkv.swapaxes(0,1).reshape(-1, hidden_size) #-> shape: [max_len, hidden_size]
        
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        print(attention)
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x.swapaxes(1,0) #-> shape: [num_attention_heads, max_len, attetion_head_size]
        # print(x.shape())
        return x

    def feed_forward(self,
                        x,
                        intermediate_weight,
                        intermediate_bias,
                        output_weight,
                        output_bias,
                    ):
        # -> shape: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # ->shape: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    def pooler_output_layer(self,x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)  # 映射到(-1,1)
        return x

    # MSE 归一化
    def layer_norm(self, x, w, b):
        # w shape:

        # x shape: [max_len, hidden_size]
        # 需要按第二维（hidden_size）归一化，与第一维（max_len）无关
        x = (x - np.mean(x, axis=1, keepdims = True)) / np.std(x, axis = 1, keepdims = True)
        print(x.shape)
        print(w.shape)
        x = np.multiply(x, w) + b  # 点乘，对位相乘
        return x

    def forward(self, x):
        x = self.embedding_forward(x)

        sequence_output = self.all_gptj_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])  ##??
        return sequence_output, pooler_output
        
        
db = DiyGptj(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)

# torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print()
