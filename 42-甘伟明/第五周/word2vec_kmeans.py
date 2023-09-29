#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json

import gensim.models
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    # model = Word2Vec.load(path)
    model = gensim.models.Word2Vec.load(path)
    return model


def load_sentence(path): # 返回分好词的句子，词与词之间由空格隔开
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 每一句话所有词的向量相加求平均，作为句子向量
        # model.wv['我们'] -> 直接返回这个词的词向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    # n_clusters =
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算
    inner_class_distance = {}
    for i in range(n_clusters):
        inner_class_distance[i] = []
    sentence_label_dict = defaultdict(list)
    for i, (sentence, label) in enumerate(zip(sentences, kmeans.labels_)):  # 取出句子和标签
        # 获取数据对应类的中心点
        center = kmeans.cluster_centers_[label]
        # print(np.array(center).shape)
        # 计算距离
        distance = vectors[i] - center
        # 将该距离添加到dict中
        inner_class_distance[label].append(distance)

        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    class_distance = []
    for data in inner_class_distance:
        class_distance.append(np.array(inner_class_distance[data]).mean(axis=0))
    print(np.array(class_distance).shape)
    # print(np.max(np.array(class_distance), axis=0))
    print(np.max(np.array(class_distance), axis=0).shape)

if __name__ == "__main__":
    main()
















