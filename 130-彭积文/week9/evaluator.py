import re
from collections import defaultdict

import numpy as np
import torch


class Evaluator:

    def __init__(self,config,model,data_loader):
        self.__config = config
        self.__model = model
        self.__data_loader = data_loader
        self.__stats_dict = {"correct":0,"wrong":0}
        self.__init_type()
        self.__type_pattern = {"LOCATION": "(04+)",
                               "ORGANIZATION": "(15+)",
                               "PERSON": "(26+)",
                               "TIME": "(37+)"}

    def __init_type(self):
        self.__type_embedding = [self.__data_loader.dataset.encoding(key) for key in self.__data_loader.dataset.get_type_dict()]

    def eval(self,epoch):
        print(f"开始测试第{epoch}轮模型效果：")
        self.__model.eval()
        # 清空上一轮结果
        self.__stats_dict = {"LOCATION": defaultdict(int),
                             "TIME": defaultdict(int),
                             "PERSON": defaultdict(int),
                             "ORGANIZATION": defaultdict(int)}

        for sentences,labels,sentences_def in self.__data_loader:
            with torch.no_grad():
                y_pred = self.__model(sentences)
            self.__write_stats(sentences_def,y_pred,labels)

        acc = self.__show_stats()
        return acc

    def __write_stats(self, sentences,y_preds,labels):
        # assert len(sentences) == len(y_preds) == len(labels)

        for sentence,y_pred,labels in zip(sentences,y_preds,labels):
            y_pred = torch.argmax(y_pred,dim=-1)
            true_entities = self.__classify_stats(sentence,labels.cpu().detach().tolist())
            pred_entities = self.__classify_stats(sentence,y_pred.cpu().detach().tolist())

            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in list(self.__type_pattern.keys()):
                self.__stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.__stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.__stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def __show_stats(self):
        F1_scores = []
        for key in list(self.__type_pattern.keys()):
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.__stats_dict[key]["正确识别"] / (1e-5 + self.__stats_dict[key]["识别出实体数"])
            recall = self.__stats_dict[key]["正确识别"] / (1e-5 + self.__stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            print("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))

        print("Macro-F1: %f" % np.mean(F1_scores))

        correct_pred = sum([self.__stats_dict[key]["正确识别"] for key in list(self.__type_pattern.keys())])
        total_pred = sum(
            [self.__stats_dict[key]["识别出实体数"] for key in list(self.__type_pattern.keys())])
        true_enti = sum([self.__stats_dict[key]["样本实体数"] for key in list(self.__type_pattern.keys())])

        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

        print("Micro-F1 %f" % micro_f1)
        print("--------------------")

    def __classify_stats(self,sentence,labels):
        labels = "".join([str(label) for label in labels[1:len(sentence) if len(sentence) < self.__config["sentence_max_length"]-1 else self.__config["sentence_max_length"]-1]])
        results = defaultdict(list)

        for key in list(self.__type_pattern.keys()):
            for location in re.finditer(self.__type_pattern[key], labels):
                start, end = location.span()
                results[key].append(sentence[start:end])

        return results