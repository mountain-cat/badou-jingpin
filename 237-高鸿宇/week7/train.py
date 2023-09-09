import torch
from model import get_net
from getdata import Dataset
from torch.utils.data import DataLoader
from config import opt
from tqdm import tqdm
import numpy as np
import os

def train(net, opt, optimizer, train_iter, valid_iter):
    '''
    分类网络训练函数

    args:
        net(nn.Module): 需要训练的网络
        opt(object): 参数配置信息
        optimizer: 优化器
    '''

    net.to(opt.device)
    print('training on', opt.device)
    min_val_loss = torch.inf

    for epoch in range(opt.num_epoch):
        watch_loss = []
        net.train()
        for x, y in tqdm(train_iter, 'epoch:'+str(epoch+1)):
            x, y = x.to(opt.device), y.to(opt.device)
            optimizer.zero_grad()
            loss = net(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f'epoch:{epoch+1}, loss:{np.mean(watch_loss)}')
        
        net.eval()
        watch_loss = []
        for x, y in tqdm(valid_iter, 'epoch:'+str(epoch+1)):
            x, y = x.to(opt.device), y.to(opt.device)
            with torch.no_grad():
                loss = net(x, y)
                watch_loss.append(loss.item())
        current_loss = np.mean(watch_loss)
        print(f'epoch:{epoch+1}, loss:{np.mean(watch_loss)}')

        if min_val_loss > current_loss:
            torch.save(net.state_dict(), os.path.join(opt.weight_save_path, 'model.pth'))
            print(f'valid loss improved from {min_val_loss} to {current_loss}')
            min_val_loss = current_loss


if __name__ == '__main__':
    train_set = Dataset(opt, is_train=True)
    valid_set = Dataset(opt, is_train=False)
    train_iter = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False)
    net = get_net(opt.hidden_size, opt.n_classes)
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    train(net, opt, optimizer, train_iter, valid_iter)