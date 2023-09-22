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


vector_size  = 100;
#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    # 分词后的每行字符串，放入set()
    sentences = set() # {'彩民 周刊 专家组 双色球 09029 期 ： 防一区 弱   三区 热', '专家 解答 ： 通过 “ 双 录取 ” 可 读 何种 名校', ...
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences: # '南海舰队 司令 登 岛礁 检查 部队 战备   菲称 苗头 不 对'
        words = sentence.split()  #sentence是分好词的，空格分开 ： ['四大', '经典', '胎教', '法', '什么', '时候', '开始', '胎教']
        vector = np.zeros(model.vector_size) # 100维全0向量
        vector_size = model.vector_size
        #所有词的向量相加求平均，作为句子向量
        for word in words: # 南海舰队
            try:
                vector += model.wv[word] # 词，返回词向量
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words)) # 一句话的累加词向量 除以 词个数，放入返回的np.array里
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    # [[-0.15490893  0.2279559  -0.02212039 ...  0.23468943 -0.09360183,   0.2159754 ], [-0.14795631  0.08143186  0.40312179 ...  0.05402408  0.12285135,   0.17269374], [-0.2927731   0.02531278  0.21290847 ...  0.058404    0.11072411,   0.22417384], ..., [-0.430
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量，句子总数取平方根，结果为聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 所有词向量按lable分类: {'0': {[],[]...}, '1': {[],[]...}}
    lable_vc_dict = {}
    for i in range(n_clusters):
        lable_i_vc = []
        for l, v in zip(kmeans.labels_, vectors):
            if i == l:
                lable_i_vc.append(v)
        lable_vc_dict[i] = lable_i_vc

    # 每个lable的平均类内距离
    lable_center_avg_distant = {} # {0: 0.2123, 1: 0.1233 ...}
    index = 0
    for center in kmeans.cluster_centers_:
        avg_distant = 0.0
        for vc in lable_vc_dict[index]:
            avg_distant += np.sqrt(np.sum((vc - center)**2)) # 离质心距离，累加
        lable_center_avg_distant[index] = avg_distant / len(lable_vc_dict[index]) # 累加的质心距离求平均
        index += 1
    # 排序：最短平均类内距离在前
    lable_center_avg_distant_sorted = sorted(lable_center_avg_distant.items(),
                                      key=lambda x: x[1], reverse=False)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    class_num = 10 # 打印10份聚类结果
    print("---类内距离最近%d类------" % class_num)
    for i in range(min(class_num, len(lable_center_avg_distant_sorted))):
        print("\n---------%d/%d lable: %d 类内平均距离：%f" % (i+1, class_num, lable_center_avg_distant_sorted[i][0], lable_center_avg_distant_sorted[i][1]))
        for t in zip(sentence_label_dict[lable_center_avg_distant_sorted[i][0]]):
            print(t[0].replace(" ", ""))

if __name__ == "__main__":
    main()

