import jieba
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertModel, BertTokenizer

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, hidden_dim, output_dim) -> None:
        super().__init__()
        self.rnn = rnn_layer
        self.cls = nn.Linear(hidden_dim, output_dim)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, x, y=None):
        x, _ = self.rnn(x)
        y_pred = self.cls(x)
        if y is not None:
            loss = self.loss_func(y_pred.view(-1, 2), y.view(-1))
            return loss
        else:
            return y_pred

class Dataset:
    def __init__(self, corpus_path, max_length) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.data = []
        self.load()

    def load(self):
        with open(self.corpus_path, encoding='utf-8') as f:
            for line in f:
                sequence = sentence_to_sequence(self.tokenizer, line)
                label = sentence_to_label(line)
                sequences, labels = self.padding(sequence, label)
                for sequence, label in zip(sequences, labels):
                    sequence = torch.LongTensor(sequence)
                    label = torch.LongTensor(label)
                    self.data.append([sequence, label])
    
    def padding(self, sequence, label):
        '''
        根据max_length对句子进行截取和padding
        '''
        sequences_with_padding, labels_with_padding = list(), list()
        sequences = [sequence[i:i+self.max_length] for i in range(0, len(sequence), self.max_length)]
        labels = [label[i:i+self.max_length] for i in range(0, len(label), self.max_length)]
        sequences_with_padding = [sequence+[0]*(self.max_length - len(sequence)) for sequence in sequences]
        labels_with_padding = [label+[-100]*(self.max_length - len(label)) for label in labels]
        return sequences_with_padding, labels_with_padding
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

def sentence_to_sequence(tokenizer, sentence:str):
    '''
    将语句转换为序列, 为后续embedding使用

    args:
        sentence(str): 需要转换的句子
    '''
    sequence = tokenizer.encode(sentence)
    return sequence[1:-1]

def sentence_to_label(sentence:str):
    '''
    生成label

    args:
        sentence(str): 需要转换的句子
    '''
    # 使用jieba分词对句子进行切分
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)    # label为1的时候说明当前字符为一个词的结尾
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer-1] = 1
    return label

def get_net(hidden_size, output_dim):
    rnn = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
    net = RNNModel(rnn, hidden_size, output_dim)
    return net

def train(net, train_iter, optmizer, num_epoch, device):
    for epoch in range(num_epoch):
        watch_loss = []
        net.to(device)
        net.train()
        for x, y in tqdm(train_iter, 'epoch:'+str(epoch+1)):
            x, y = x.to(device), y.to(device)
            optmizer.zero_grad()
            loss = net(x, y)
            loss.backward()
            optmizer.step()
            watch_loss.append(loss.item())
        print(f'epoch:{epoch+1}, loss:{np.mean(watch_loss)}')
    return net

def predict(net, input_strings, device):
    net.to(device)
    net.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for string in input_strings:
        seq = sentence_to_sequence(tokenizer, string)
        with torch.no_grad():
            x = torch.LongTensor([seq])
            x = x.to(device)
            result = net(x)[0]
            result = torch.argmax(result, dim=1)
            for index, p in enumerate(result):
                if p == 1:
                    print(string[index], end=' ')
                else:
                    print(string[index], end='')
            print()

if __name__ == "__main__":
    corpus_path = 'workspace/badouai/week6/corpus.txt'
    max_length = 20
    batch_size = 1024
    hidden_size = 768
    output_dim = 2
    lr = 1e-5
    num_epoch = 10
    device = torch.device('cuda:0')
    dataset = Dataset(corpus_path, max_length)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    net = get_net(hidden_size, output_dim)
    print(net)
    optmizer = torch.optim.Adam(net.parameters(), lr=lr)
    trained_net = train(net, train_iter, optmizer, num_epoch, device)

    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict(net, input_strings, device)