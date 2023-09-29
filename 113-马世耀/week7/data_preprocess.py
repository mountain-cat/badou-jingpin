#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-09-01 15:15:03
LastEditors: Shiyao Ma
LastEditTime: 2023-09-01 15:28:26
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
import os
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
CURR_DIR = os.path.dirname(__file__)
print(CURR_DIR)


def load_data(data_path = os.path.join(CURR_DIR, '文本分类练习.csv')):
    data = pd.read_csv(data_path, header = 0)
    print(data.shape)
    print(data.head(1))
    return data

def main():
    train_size = Config['train_size']
    data = load_data()
    train, valid = train_test_split(data, train_size = train_size, random_state = Config['seed'])
    print(f"average sentence length in train: {train['review'].map(lambda ele: len(ele)).mean():.2f}, q90: {train['review'].map(lambda ele: len(ele)).quantile(0.9):.2f}")
    print(f"average sentence length in valid: {valid['review'].map(lambda ele: len(ele)).mean():.2f}, q90: {valid['review'].map(lambda ele: len(ele)).quantile(0.9):.2f}")
    print(f"saving train {train.shape[0]} and valid {valid.shape[0]} to files...")
    train.to_json(os.path.join(CURR_DIR, 'train.json'), orient = 'records', force_ascii = False, lines = True)
    valid.to_json(os.path.join(CURR_DIR, 'valid.json'), orient = 'records', force_ascii = False, lines = True)



if __name__ == "__main__":
    main()