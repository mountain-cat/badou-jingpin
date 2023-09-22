### 简介
随机10个字符为1个字符串，如果e,d其中一个字符存在字符串里就是正样本；反之则是负样本
通过RNN去进行预测

##### 模型初始化
```python 
 def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

``` 
由于这个是二分类问题，所以试用了sigmoid和均方差实现

##### 模型预测过程
```python 
 def forward(self, x, y=None):
      x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
      rnn_output, _ = self.rnn(x)  # RNN处理 (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
      x = rnn_output[:, -1, :]  # 取最后一个时刻的输出 (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
      x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 1)
      y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
      if y is not None:
          return self.loss(y_pred, y)  # 预测值和真实值计算损失
      else:
          return y_pred  # 输出预测结果
```

#####  样本生成方法
```python 
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = realOutput(x)

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def realOutput(x):
    # 指定哪些字出现时为正样本
    if set("ed") & set(x):
        y = 1
        # 指定字都未出现，则为负样本
    else:
        y = 0
    return y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)
``` 

#####  配置参数
```python 
# 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.005  # 学习率
```

##### 模型训练图表
![image](https://github.com/koklinliau/badou-jingpin/assets/140817016/1d25a54a-ad89-4aa1-b69f-937106e531e1)


