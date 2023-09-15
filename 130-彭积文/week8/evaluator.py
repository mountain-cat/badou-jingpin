import torch


class Evaluator:

    def __init__(self,config,model,data_loader):
        self.__config = config
        self.__model = model
        self.__data_loader = data_loader
        self.__stats_dict = {"correct":0,"wrong":0}
        self.__init_type()

    def __init_type(self):
        self.__type_embedding = [self.__data_loader.dataset.encoding(key) for key in self.__data_loader.dataset.get_type_dict()]

    def eval(self,epoch):
        print(f"开始测试第{epoch}轮模型效果：")
        self.__model.eval()
        self.__stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果

        # 获取类型的embedding
        types_normal = self.__get_type_normalize()

        for sentences,types_id in self.__data_loader:
            with torch.no_grad():
                sentences_embedding = self.__model(sentences)
            self.__write_stats(sentences_embedding,types_id,types_normal)
        acc = self.__show_stats()
        return acc

    def __get_type_normalize(self):
        return torch.nn.functional.normalize(self.__model(torch.stack([type for type in self.__type_embedding], dim=0).to(self.__config["device"])),dim=-1)

    def __write_stats(self, sentences_embedding,types_id,type_embedding):
        assert len(sentences_embedding) == len(types_id)
        for sentence_embedding,type_id in zip(sentences_embedding,types_id):
            mm = torch.mm(sentence_embedding.unsqueeze(0),type_embedding.T) #矩阵乘法
            pred_index = int(torch.argmax(mm.squeeze()))
            if type_id == pred_index:
                self.__stats_dict["correct"] += 1
            else:
                self.__stats_dict["wrong"] += 1

    def __show_stats(self):
        correct = self.__stats_dict["correct"]
        wrong = self.__stats_dict["wrong"]
        print(f"预测集合条目总量：{correct + wrong}")
        print(f"预测正确条目：{correct}，预测错误条目：{wrong}")
        print(f"预测准确率：{correct / (correct + wrong)}")
        print("--------------------")
        return correct / (correct + wrong)