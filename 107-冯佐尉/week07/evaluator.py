import torch


class Evaluator:

    def __init__(self,config,model,data_loader):
        self.__config = config
        self.__model = model
        self.__data_loader = data_loader
        self.__stats_dict = {"correct":0,"wrong":0}

    def eval(self,epoch):
        print(f"开始测试第{epoch}轮模型效果：")
        self.__model.eval()
        self.__stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for x,y in self.__data_loader:
            with torch.no_grad():
                y_pred = self.__model(x)  # 不输入labels，使用模型当前参数进行预测
            self.__write_stats(y, y_pred)
        acc = self.__show_stats()
        return acc

    def __write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.__stats_dict["correct"] += 1
            else:
                self.__stats_dict["wrong"] += 1
        return

    def __show_stats(self):
        correct = self.__stats_dict["correct"]
        wrong = self.__stats_dict["wrong"]
        print(f"预测集合条目总量：{correct + wrong}")
        print(f"预测正确条目：{correct}，预测错误条目：{wrong}")
        print(f"预测准确率：{correct / (correct + wrong)}")
        print("--------------------")
        return correct / (correct + wrong)