#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def calc_distance1(sentence_label_vec, cluster_centers):
    """
    欧式距离
    :return:
    """
    label_distance = dict()
    for label, vec_list in sentence_label_vec.items():
        center = cluster_centers[label]
        distance = 0
        for vec in vec_list:
            distance += math.sqrt(sum((np.array(vec) - np.array(center)) ** 2))
        label_distance[label] = round(distance/len(vec_list), 2)
    label_distance = dict(sorted(label_distance.items(), key=lambda x: x[1]))
    return label_distance


def calc_distance2(sentence_label_vec, cluster_centers):
    """
    夹角的余弦值（余弦相似度）越接近1越相似；
    余弦距离=1-夹角余弦值
    :return:
    """
    label_distance = dict()
    for label, vec_list in sentence_label_vec.items():
        center = cluster_centers[label]
        distance = 0
        for vec in vec_list:
            distance += np.dot(vec, center) / (np.sqrt(np.sum(vec * vec)) * np.sqrt(np.sum(center * center)))
        label_distance[label] = round(distance/len(vec_list), 2)
    label_distance = dict(sorted(label_distance.items(), key=lambda x: x[1], reverse=True))
    return label_distance



def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    sentence_label_dict = defaultdict(list)
    sentence_label_vec = defaultdict(list)

    for vector, sentence, label in zip(vectors, sentences, kmeans.labels_):  # 取出句子和类别 min=0, max=41
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        sentence_label_vec[label].append(vector)  # 同标签的向量放到一起

    # 计算类内平均距离，将距离最小的类优先输出
    # label_distance = calc_distance1(sentence_label_vec, kmeans.cluster_centers_)
    label_distance = calc_distance2(sentence_label_vec, kmeans.cluster_centers_)
    print(label_distance)
    for index, label in enumerate(label_distance.keys()):
        content = sentence_label_dict[label]
        print("cluster %s :" % label)
        for i in range(min(10, len(content))):  # 随便打印几个，太多了看不过来
            print(content[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
