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

def eculid_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2),axis=1)).mean()

def calc_distance(vectors_dict, cluster_centers):
    distance_label_dict = dict()
    for label, vectors in vectors_dict.items():
        vectors = np.array(vectors)
        center = cluster_centers[label]
        distance = eculid_distance(center, vectors)
        distance_label_dict[label] = distance
    return distance_label_dict

def main():
    model = load_word2vec_model("week5/model.w2v") #加载词向量模型
    sentences = load_sentence("week5/titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  #取出句子, 对应的向量, 和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        vector_label_dict[label].append(vector)         #同标签的放到一起
    
    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_
    distances = calc_distance(vector_label_dict, cluster_centers)

    for label, distance in distances.items():
        print("cluster %s avg distance %f:" % (label, distance))
        sentences = sentence_label_dict[label]
        for i in range(min(5, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

