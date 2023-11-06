from  SegDataset1 import SegDataset
from  orther import orther
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm
from loss import loss

num_epochs=80
torch.cuda.init()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=SegDataset('training.txt')
train_loader = DataLoader(dataset, batch_size=16,shuffle=True, drop_last=True)
model=orther().to(device)
# model.load_state_dict(torch.load("model5.pth"))

optimizer = optim.SGD(model.parameters(), lr=2e-6, momentum=0.9)
crf_loss = loss().to(device)


for epoch in range(num_epochs):
    all=0
    acc=0
    for idx, batch_samples in enumerate(tqdm(train_loader)): 
        optimizer.zero_grad()
        text,data,tab=batch_samples
        data = torch.squeeze(data)
        data, tab = data.to(device), tab.to(device)  # 将数据移到GPU上
        # print(tab[0])
        # print("----------------------------------")
        pre=model(data)
        loss=crf_loss(pre,tab)
        max_indices = torch.argmax(pre, dim=2)
        # print(max_indices[0])
        correct_predictions = (max_indices == tab).sum()
        acc=acc+correct_predictions
        all=all+ pre.size(0) * 256
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'model'+str(epoch)+'.pth')
    print(loss.item())
    print(acc/all)
    print('------------',epoch,'-------------')