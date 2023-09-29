from  SegDataset1 import SegDataset
from  orther import orther
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=5
dataset=SegDataset('training.txt')
train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, drop_last=True)
model=orther().to(device)
model.load_state_dict(torch.load("model54.pth"))

for idx, batch_samples in enumerate(tqdm(train_loader)): 
    text,data,t=batch_samples
   #  print(t[0])
    data = data.to(device) # 将数据移到GPU上
    data = torch.squeeze(data)

    pre=model(data) #(batch,lo24,4)
    max_indices = torch.argmax(pre, dim=2)
   #  print(max_indices[0])
    for i in range(batch_size):
      result=[]
      sentence=""
      for j in range(len(text[i])):
         if(j<256): t=pre[i][j+1]
         else : t=pre[i][j]
         result.append(text[i][j] )
         if torch.argmax(t, dim=-1) == 2 or torch.argmax(t, dim=-1)==3 :
            result.append("  " )

      sentence = " ".join(result)
      print(sentence)
