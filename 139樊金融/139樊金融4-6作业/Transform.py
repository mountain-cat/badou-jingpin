import torch.nn as nn
import torch

import copy


class Transform(nn.Module):
    def __init__(self,batch, N,E_N,device):
        super(Transform, self).__init__()
        self.device        = device
        self.embedding     = nn.Embedding(7000, 256).to(self.device)
        self.EncoderList   = nn.ModuleList([Encoder(batch=batch,N=E_N).to(self.device) for _ in range(N)])    
        self.c             =nn.Linear(256,256)
        self.c1            =nn.Linear(128,4)
        self.c2            =nn.Linear(256,128)
        self.bn1           = nn.BatchNorm1d(256)
        self.lrelu         = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = LayerNorm((batch,256,256))

        self.batch         =batch

    def forward(self, x):#x输入为（bitch,256）
        size = (self.batch,256)
        result_p = torch.ones(*size) * torch.arange(1, 256 + 1)#(batch,256)0-256
        result_p =result_p.to(self.device).long()
        x=self.embedding(x)
        t_position=self.embedding(result_p)
        x=self.norm((x+t_position))

        for module in self.EncoderList:
            x=self.norm(module(x)+x)
        x=self.lrelu(self.bn1(self.c(x)))
        x=self.lrelu(self.bn1(self.c2(x)))
        x=self.bn1(self.c1(x))
        return x



class Encoder(nn.Module):
    def __init__(self,batch, N):
        super(Encoder, self).__init__()
        self.norm = LayerNorm((batch,256,256))
        self.mul_attention= [
            Attention(batch=batch,b=16) for _ in range(N)
                ]
        self.c=nn.Linear(256*N,512)
        self.c1=nn.Linear(512,256)
        self.c2=nn.Linear(128,256)
        self.c4=nn.Linear(128,128)
        self.c3=nn.Linear(256,128)
        self.c5=nn.Linear(512,512)
        self.bn1 = nn.BatchNorm1d(256)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self,x):
        r_attention = [model(x) for model in self.mul_attention]#x-》attion ，n*x
        r_attention = torch.cat(r_attention, dim=-1) #(b,256,256*n)
        r1=self.lrelu(self.bn1(self.c(r_attention)))
        r1=self.lrelu(self.bn1(self.c5(r1)))
        r1=self.norm(self.lrelu(self.c1(r1))+x)

        r=self.lrelu(self.bn1(self.c3(r1)))
        r=self.lrelu(self.bn1(self.c4(r)))
        r=self.lrelu(self.bn1(self.c2(r)))
        return self.norm(r+r1)
        
        
        


class Attention(nn.Module):
    def __init__(self,batch,b):
        super(Attention,self).__init__()
        self.c1=nn.Linear(256,256)
        self.c2=nn.Linear(256,256)
        self.c3=nn.Linear(256,256)
        self.c4=nn.Linear(256,256)
        self.lrelu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.b =b
        

    def forward(self, x):#x输入为(batch,256,256)
        x1=self.lrelu(self.bn1(self.c1(x)))
        x2=self.lrelu(self.bn1(self.c2(x)))
        x3=self.lrelu(self.bn1(self.c3(x)))#(b,256,256)
        r=torch.matmul(x1, x2.transpose(1, 2))/(self.b)#(b,256,256)
        r=torch.softmax(r,-1)
        r=torch.matmul(r,x3)#(b,256,256)
        r=self.lrelu(self.bn1(self.c4(r)))
        return r


    





class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        t= (x - mean) 
        t=self.a_2 *t
        t=t/ (std + self.eps)
        r=t + self.b_2
        return r
        # return  (x - mean) / (std + self.eps) 
        