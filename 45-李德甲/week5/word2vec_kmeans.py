#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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

def cal_avg_distince(center_point,plist):
    s=len(plist)
    sum=0
    for p in plist:
        sum+=cal_distince(np.array(center_point),np.array(p))
    return sum/s

def cal_distince(v1,v2):
    return np.linalg.norm(v1-v2)

def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model
def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    # sentences = []
    # with open("corpus.txt", encoding="utf8") as f:
    #     for line in f:
    #         sentences.append(jieba.lcut(line))
    # model = train_word2vec_model(sentences, 100)

    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    label_vector_dict = defaultdict(list)
    label_distince_dict={}
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for vector, label in zip(vectors, kmeans.labels_):  #取出句子向量和标签
        label_vector_dict[label].append(vector)         #同标签的放到一起
    for label, plist in label_vector_dict.items():#计算平均距离
        avg_distince=cal_avg_distince(kmeans.cluster_centers_[label],plist)
        label_distince_dict[label]=avg_distince
    #按照平均距离排序
    sorted_items=sorted(label_distince_dict.items(),key=lambda x:x[1])

    for label_distince in sorted_items:
        label=label_distince[0]
        distince=label_distince[1]
        print("cluster %s ,平均距离 %.5f:" % (label,distince))
        label_sentences=sentence_label_dict[label]
        for i in range(min(10, len(label_sentences))):  #随便打印几个，太多了看不过来
            print(label_sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

