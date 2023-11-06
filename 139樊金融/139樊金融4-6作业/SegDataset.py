from transformers import BertTokenizer
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn


class SegDataset(Dataset):
    def __init__(self,src):
        self.tdict=dict()
        self.textData=[]
        self.trainData=[]
        self.max=0
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
        line =torch.zeros(256)#第一个维度是字 第二个维度是向量
        label=torch.zeros(256)#lable分为开始 结束 中间段 独立
        for i in range(256):
            if(i >= len(self.trainData[idx])):
                line[i]=self.tdict['padding']
            else :
                line[i]=self.tdict[self.trainData[idx][i]]
        index=0 
        text=self.trainData[idx]
        flag=0   #flag表示前面有没有起始位  
        for i in range(256) :
            
            if(i >= len(self.trainData[idx])):
                    label[i]=3
            else : 
                if(index==len(self.textData[idx])-1 and flag==1 ):
                    label[i]=1
                    flag=0
                elif(index==len(self.textData[idx])-1 and flag==0 ):
                    label[i]=3
                    flag=0  
                elif(index==0 and self.textData[idx][index+1]!=' '):
                    label[i]=0
                    flag=1
                elif(index==0 and self.textData[idx][index+1]==' '):
                    label[i]=3
                    index=index+2
                elif(index>=256 and flag==1) :
                    label[i]=1
                    flag=0
                elif(index>=256 and flag==0) :
                    label[i]=3
                    flag=0
                elif(self.textData[idx][index+1]==' 'and flag==1):

                    label[i]=1
                    index=index+2
                    flag=0

                elif(self.textData[idx][index+1]==' 'and flag==0):
                    label[i]=3
                    index=index+2
                elif(index==0):
                    label[i]=0
                    flag=1
                elif(self.textData[idx][index-1]==' 'and flag==0):
                    label[i]=0
                    flag=1
                elif(self.textData[idx][index-1]!=' 'and flag==1):
                    label[i]=2
                else:
                    label[i]=3
                index+=1 




        return [text,line.long(), label.long()]

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