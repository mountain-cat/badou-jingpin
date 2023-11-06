import torch
import torch.nn as nn


embedding = nn.Embedding(5000, 256) 
tdict=dict()
textData=[]
trainData=[]
max=0
count=0
with open('output.txt', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        linex=line.replace(" ", "")

        if(len(linex) > 128):
            max +=1 
        trainData.append(linex)
        textData.append(line)#有空格
        for j, word in enumerate(line):
         if word not in tdict : 
            count+=1  
            tdict[word]=count
tdict['UNKNOW']=count+1
tdict['padding']=count+2    
idx=0  
line =torch.empty(1024)#第一个维度是字 第二个维度是向量
label=torch.empty((1024, 4))#lable分为开始 结束 中间段 独立
for i in range(1024):
    if(i >= len(trainData[idx])):
       line[i]=tdict['padding']
    else :
        line[i]=tdict[trainData[idx][i]]
index=0 
flag=0   #flag表示前面有没有起始位  
for i in range(1024) :
    
    if(i >= len(trainData[idx])):
        label[i]=torch.zeros(4)
        label[i][3]=1
    else : 
        if(index==len(textData[idx])-1 and flag==1 ):
            label[i]=torch.zeros(4)
            label[i][1]=1
            flag=0
         
        elif(index==len(textData[idx])-1 and flag==0 ):
            label[i]=torch.zeros(4)
            label[i][3]=1
            flag=0    
        elif(textData[idx][index+1]==' 'and flag==1):
            
            label[i]=torch.zeros(4)
            label[i][1]=1
            index=index+2
            flag=0
            
        elif(textData[idx][index+1]==' 'and flag==0):
            label[i]=torch.zeros(4)
            label[i][3]=1
            index=index+2
        elif(index==0):
            label[i]=torch.zeros(4)
            label[i][0]=1
            flag=1
        elif(textData[idx][index-1]==' 'and flag==0):
            label[i]=torch.zeros(4)
            label[i][0]=1
            flag=1
        elif(textData[idx][index-1]!=' 'and flag==1):
            label[i]=torch.zeros(4)
            label[i][2]=1
        else:
            label[i]=torch.zeros(4)
            label[i][3]=1
        index+=1   
print(line)
