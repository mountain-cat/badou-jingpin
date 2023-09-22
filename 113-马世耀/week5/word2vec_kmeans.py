#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-08-17 14:43:26
LastEditors: Shiyao Ma
LastEditTime: 2023-08-17 17:52:28
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import os
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
print(f"working under {os.path.dirname(os.path.abspath(__file__))}")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    sentence_distance = {k: [] for k in range(n_clusters)}

    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        curr_center = kmeans.cluster_centers_[label]
        sentence_distance[label].append(((vector - curr_center) ** 2).sum() ** 0.5)
    sentence_distance = {k: np.mean(v) for k, v in sentence_distance.items()}
    sentence_distance = dict(sorted(sentence_distance.items(), key = lambda ele: ele[1], reverse = False))

    # for label, sentences in sentence_label_dict.items():
    for label, dist in sentence_distance.items():
        print(f"cluster {label}, with inner-cluster simi of {dist:.4f} :")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

