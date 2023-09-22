
import numpy as np
from torch.utils.data import Dataset, DataLoader


def build_sample():
    x = np.random.random(5)
    if x[0] > x[4] and x[1] > x[2]:
        return x, 0
    elif x[0] > x[4] and x[1] < x[2]:
        return x, 1
    elif x[0] < x[4] and x[1] > x[2]:
        return x, 2
    elif x[0] < x[4] and x[1] < x[2]:
        return x, 3


def build_dataset(total_sample_num):
    X0, X1, X2, X3 = [], [], [], []
    Y0, Y1, Y2, Y3 = [], [], [], []
    sample0_num, sample1_num, sample2_num, sample3_num = 0, 0, 0, 0
    for i in range(10000000):
        x, y = build_sample()
        if y == 0:
            sample0_num += 1
            X0.append(x)
            Y0.append([y])
        elif y == 1:
            sample1_num += 1
            X1.append(x)
            Y1.append([y])
        elif y == 2:
            sample2_num += 1
            X2.append(x)
            Y2.append([y])
        elif y == 3:
            sample3_num += 1
            X3.append(x)
            Y3.append([y])
        if np.min((sample0_num, sample1_num, sample2_num, sample3_num)) >= (total_sample_num//4):
            break

    x0, y0 = np.array(X0), np.array(Y0)
    x1, y1 = np.array(X1), np.array(Y1)
    x2, y2 = np.array(X2), np.array(Y2)
    x3, y3 = np.array(X3), np.array(Y3)

    min_num = total_sample_num // 4
    x = (np.array([x0[:min_num, :], x1[:min_num,:], x2[:min_num,:], x3[:min_num,:]]).reshape(-1, 5)).astype(np.float32)
    y = np.array([y0[:min_num, :], y1[:min_num, :], y2[:min_num, :], y3[:min_num, :]]).reshape(-1, 1)
    return x, y


class ReadDataFromFile(Dataset):
    def __init__(self, image_arr, label_arr):
        self.data_arr = np.asarray(image_arr)
        self.label_arr = np.asarray(label_arr)
        self.data_len = self.data_arr.shape[0]

    def __getitem__(self, index):
        mean = np.mean(self.data_arr[index], axis=0, keepdims=True)
        std = np.std(self.data_arr[index], axis=0, keepdims=True)
        single_img_tensor = (self.data_arr[index] - mean) / std
        single_image_label = self.label_arr[index]
        return single_img_tensor, single_image_label

    def __len__(self):
        return self.data_len


def get_data(total_num=2000, batch_size=16):
    x, y = build_dataset(total_num)
    train_data = ReadDataFromFile(image_arr=x, label_arr=y)
    train_dataset_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_dataset_loader, x.shape[0]/batch_size
