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
    vectorsDirc = {}
    vectors = []
    i = 0
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
        vectorsDirc[i] = sentence
        vectors.append(vector / len(words))
        i += 1
    return np.array(vectors), vectorsDirc


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors, vectorsDirc = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentenceVector, label in zip(vectorsDirc, kmeans.labels_):
        sentence_label_dict[label].append(sentenceVector)
    # for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
    #     sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        similar = {}
        for i in range(len(sentences)):
            similar[sentences[i]] = math.sqrt(sum((x - y) ** 2 for x, y in zip(vectors[label], vectors[sentences[i]])))
            # similar[sentences[i]] = pow(pow(vectors[label] - vectors[sentences[i]], 2), 0.5)
        similar = sorted(similar.items(), key=lambda x: x[1])
        for i in range(min(10, len(similar))):
            print(f'聚类相似度最强的句子是： {vectorsDirc[similar[i][0]]}, 相似度: {similar[i][1]}')
        print("---------")


def calculate_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("向量维度不一致")

    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(vector1, vector2)))
    return distance


if __name__ == "__main__":
    main()
