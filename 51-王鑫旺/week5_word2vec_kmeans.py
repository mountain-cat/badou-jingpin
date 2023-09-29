#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
import pandas as pd
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

def calculate_distance(sentences,vectors,kmeans):
    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    distance_label_dict=defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for vector, label in zip (vectors, kmeans.labels_):
        vector_label_dict[label].append(vector)
    for label, vectors in vector_label_dict.items():
        center = kmeans.cluster_centers_[label]
        distances=0
        for vector in vectors:
            distance = np.linalg.norm(vector-center)
            distances += distance
        distance_label_dict[label] = distances/len(vectors)
    merged_dict=defaultdict(dict)
    for label, dis in distance_label_dict.items():
        merged_dict[label]['distance']=dis
    for label,sen in sentence_label_dict.items():
        merged_dict[label]['sentences']=sen
    sorted_list = sorted(merged_dict.items(), key=lambda x: x[1]['distance'], reverse=True)
    
    for label, dic in sorted_list:
        print("cluster: %s" % label, "distance: %f" % dic['distance'])
        
    
    
    
    return sorted_list


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    # print(kmeans.cluster_centers_[kmeans.labels_[0]])
    
    sorted_list=calculate_distance(sentences,vectors,kmeans)
    for label, dis_and_sen in sorted_list:
        print("cluster %s :" % label)
        for i in range(min(5, len(dis_and_sen['sentences']))):  #随便打印几个，太多了看不过来
            print(dis_and_sen['sentences'][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

