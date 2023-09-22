from transformers import BertTokenizer
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction


class SegDataset(Dataset):
    def __init__(self,src):
        self.tdict=dict()
        self.textData=[]
        self.trainData=[]
        self.max=0
        self.tokenizer=BertTokenizer.from_pretrained(r"D:\deeplean\china_split\bert-base-chinese", return_dict=False)

        count=0
        with open(src, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                linex=line.replace(" ", "")
                # if(len(linex) > self.max):
                    # self.max = len(linex) 
                self.trainData.append(linex)
                self.textData.append(line)#有空格
                for j, word in enumerate(line):
                 if j<1000 and j not in self.tdict : 
                  count+=1
                  self.tdict[j]=count
                 if word not in self.tdict : 
                    count+=1  
                    self.tdict[word]=count
        self.tdict['UNKNOW']=count+1
        self.tdict['padding']=count+2            
    def __getitem__(self, idx):
        label=torch.zeros(256)#lable分为开始 结束 中间段 独立

        text=self.trainData[idx]
        text2=self.textData[idx]
        sen_code = self.tokenizer(text=text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        index=0 
        label[0]=3
        for i in range(255) :

            if(i >= len(self.trainData[idx])):
                    label[i]=3
            elif(i==0 and index==len(self.textData[idx])-1):
                label[i+1]=3        
            elif(i==0 and self.textData[idx][index+1]==' '):
                label[i+1]=3
                index=index+2
            elif(i==0):
                label[i+1]=0
                
            elif((i==254 or index==len(self.textData[idx])-1) and self.textData[idx][index-1]==' '):
                label[i+1]=3    

            elif((i==254 or index==len(self.textData[idx])-1)):
                label[i+1]=2
            elif(self.textData[idx][index-1]==' ' and self.textData[idx][index+1]==' '):
                label[i+1]=3
                index=index+2
            elif(self.textData[idx][index-1]!=' ' and self.textData[idx][index+1]==' '):
                label[i+1]=2
                index=index+2
            elif(self.textData[idx][index-1]!=' ' and self.textData[idx][index+1]!=' '):
                label[i+1]=1    
            elif(self.textData[idx][index-1]==' ' and self.textData[idx][index+1]!=' '):
                label[i+1]=0
            index=index+1  
            # tx1=self.textData[idx][index]
            # tx2=self.trainData[idx][i]
            # len1=len(self.textData[idx])  



        return [self.trainData[idx],torch.tensor(sen_code['input_ids']) , label.long()]

    def __len__(self):
         """get dataset size"""
         return len(self.textData)



# tdict=dict()
# textData=[]
# count=0
# with open('training.txt', 'r', encoding='utf-8') as file:
#     for i, line in enumerate(file):
#        textData.append(line)
#        for j, word in enumerate(line):
#          if word not in tdict : 
#             count+=1  
#             tdict[word]=count
# tdict['UNKNOW']=count+1
# print(tdict)            