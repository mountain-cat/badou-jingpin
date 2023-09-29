
'''
用bert对天池上的《零基础NLP新闻分类》任务进行分类
但是得到的结果只有0.93，排名600多
'''






#import 相关库
#导入前置依赖
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# 用于加载bert模型的分词器
from pathlib import Path
# 用于加载bert模型
from transformers import BertModel

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time

def convert_to_one_hot(label):
    one_hot = np.zeros(14)
    one_hot[label] = 1
    return one_hot
def random_sample(df):
    texts = []
    labels = []
    # print(df.head(5))
    for _, text in df.iterrows():
        texts.append(   [int(te) for te in text.text.split()] )
        labels.append(convert_to_one_hot(text.label))
        # print(text.label)
        # print(text)
    return labels, texts

def load_csv (mode='train'):
    if mode == 'train':
        train_df = pd.read_csv('./train_set.csv/train_set.csv', sep='\t')
        # print(train_df.head(5),'\n')
        # print(train_df.info(),'\n')
        # print(train_df.describe(),'\n')
        # print(len(train_df.text.iloc[1].split()),'\n')
        # print(train_df.columns,'\n')
        validation = train_df.sample(frac=0.1)
        train = train_df[~train_df.index.isin(validation.index)]
    
        train_labels, train_texts = random_sample(train)
        validation_labels, validation_texts = random_sample(validation)
        # print(train_texts[1])
       
        return train_labels, train_texts, validation_labels, validation_texts
    if mode == 'test':
        test_texts=[]
        test_df = pd.read_csv('./test_a.csv/test_a.csv', sep='\t')
        for text in test_df.iterrows():
            # print(text[1].text.split())
            test_texts.append(text[1].text.split())
        # print(test_texts[0:2])
        # print(test_df.head(5),'\n')
        # print(test_df.info(),'\n')
        # print(test_df.describe(),'\n')
        # print(test_df.shape,'\n')
        # print(test_df.columns,'\n')
        # print(test_df.index)
        return test_texts

   
   
class Mydataset(Dataset):
    def __init__(self,  texts,labels= None):
        super().__init__()
        self.labels = labels
        self.texts = texts
        # print(self.labels)
    def __len__(self):
        return len(self.texts) 
    
    def __getitem__(self, index):
        text = self.texts[index]
        if self.labels is not None:
            label = self.labels[index]
            return text, label
        else :
            return text
def collate_fn(batch):
    max_length=100
    texts, labels = zip(*batch)
    # print(labels)
    padded_texts = pad_sequence([torch.tensor(text) for text in texts],batch_first=True, padding_value=0)
    truncated_texts = padded_texts[:, :max_length]
    return torch.tensor(truncated_texts),torch.tensor(labels)
    
        
def build_dataloader (batchsize = 32):
    train_labels, train_texts, validation_labels, validation_texts = load_csv('train')
    # print(train_labels)
    train_dataset = Mydataset(train_texts,train_labels)
    validation_dataset = Mydataset(validation_texts, validation_labels)
    
    train_loader = DataLoader(train_dataset, batch_size= batchsize, shuffle= True, collate_fn= collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batchsize, shuffle= False,collate_fn= collate_fn)
    # for x, y in train_loader:
    #     print(x)
    #     print('label:',y)
    # print(train_loader)    
    
    return train_loader, validation_loader
        
class bertmodel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained('../../bert-base-uncased',return_dict=False)
        self.classify = nn.Linear(input_dim, 14)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, texts,labels=None):
        sequence_out, pooler_out = self.bert(texts)        
        # print('poolerout', type(pooler_out))

        # print(type(texts),texts.shape)
        out = self.classify(pooler_out)
        pred = self.activation(out)
        # print(pred.shape, labels.shape)
        if labels is not None:
            return self.loss(pred, labels)
        else :
            return pred
        
        
def validate(model, validation_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_loss=0
    total_correct = 0
    for text, label in validation_loader:
        text=text.to(device)
        label = label.to(device)
        loss = model(text, label)
        total_loss += float(loss)
        pred_indices = torch.argmax(model(text), dim = 1)
        true_indices = torch.argmax(label, dim =1)
        count = torch.sum(pred_indices==true_indices)
        total_correct += count.item()
        del text, label
        torch.cuda.empty_cache()
    return total_correct/len(validation_loader.dataset), total_loss/len(validation_loader.dataset)
    
        
        
        
def main():
    start = time.time()
    epoch_num = 40
    batch_size = 100
    input_dim = 768
    best_accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 清空一下cuda缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 模型存储路径
    model_dir = Path("./model/bert_checkpoints")
    # 如果模型目录不存在，则创建一个
    os.makedirs(model_dir) if not os.path.exists(model_dir) else ''
    model = bertmodel(input_dim)
    model.to(device)
    train_loader, validation_loader = build_dataloader(batch_size)
    optim = torch.optim.Adam(model.parameters(),lr = 1e-5)
    for epoch in range(epoch_num):
        model.train()
        for text, label in train_loader:
            # print(type(text),type(label))
            text=text.to(device)
            label = label.to(device)
            optim.zero_grad()
            loss = model(text, label)
            loss.backward()
            optim.step()
            print("Time {}, Epoch {}/{}, total loss:{:.4f}".format(time.time()-start, epoch+1, epoch_num, loss))
            del text, label
            torch.cuda.empty_cache()
        accuracy, validation_loss = validate(model, validation_loader)
        print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))
        torch.save(model, model_dir / f"model_{epoch}.pt")
        if accuracy >= best_accuracy:
            torch.save(model, model_dir / "model_best.pt")
        print("total time:", time.time()-start)
    return


def test():
    train_labels, train_texts, validation_labels, validation_texts = load_csv('train')
    # print(train_labels)
    train_dataset = Mydataset(train_texts,train_labels)
    validation_dataset = Mydataset(validation_texts, validation_labels)
    batchsize = 10
    train_loader = DataLoader(train_dataset, batch_size= batchsize, shuffle= True, collate_fn= collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batchsize, shuffle= False,collate_fn= collate_fn)
    # 假设有一个数据加载器 dataloader
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 获取原始数据集的数据量
    dataset_size = len(train_loader.dataset)
    print("原始数据集的数据量:", dataset_size)
            
if __name__ == "__main__":
    # load_csv()
    # load_csv('test')
    # build_dataloader()
    main()
    # test()















