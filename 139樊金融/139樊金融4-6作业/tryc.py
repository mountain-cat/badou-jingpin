from  SegDataset import SegDataset
from  Transform import Transform
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm

device = torch.device( 'cpu')
batch_size=20
dataset=SegDataset('output.txt')
train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, drop_last=True)
model=Transform(batch=batch_size, N=6,E_N=10,device=device).to(device)
model.load_state_dict(torch.load("model39.pth"))

for idx, batch_samples in enumerate(tqdm(train_loader)): 
    text,data,t=batch_samples
    print(t[0])
    data = data.to(device) # 将数据移到GPU上
    pre=model(data) #(batch,lo24,4)
    max_indices = torch.argmax(pre, dim=2)
    print(max_indices[0])
    for i in range(batch_size):
      result=[]
      sentence=""
      for j in range(len(dataset.trainData[i])):
         t=pre[i][j]
         result.append(dataset.trainData[i][j] )
         if torch.argmax(t, dim=-1) == 2 or torch.argmax(t, dim=-1)==3 :
            result.append("  " )

      sentence = " ".join(result)
      print(sentence)
