import torch
import torch.nn as nn
import numpy as np
"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：
# 0：第一个大于第六个数
# 1：第二个大于第五个数
# 2：第三个大于第四个数
# 3：其他
# 优先级从上往下，如第一个数大于第六个，第二个大于第五个，则判断为第0类
"""
class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear=nn.Linear(input_size,4)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        y_pred=self.linear(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
#训练样本生成
def build_sample():
    x=np.random.random(6)
    if x[0]>x[5]:
        return x,0
    elif x[1]>x[4]:
        return x,1
    elif x[2]>x[3]:
        return x,2
    else:
        return x,3
#训练集
def build_dataset(sample_num):
    X=[]
    Y=[]
    for i in range(sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

#用测试集测试准确率
#返回准确率
def evaluate(model):
    model.eval()
    test_sample_num=100
    X,Y=build_dataset(test_sample_num)
    # print("Test dataset contains %d class 0, %d class 1, %d class 2 samples" % (sum(Y == 0), sum(Y == 1), sum(Y == 2)))
    correct,wrong=0,0
    with torch.no_grad():
        y_pred=model(X)
    _,y_pred=torch.max(y_pred,dim=1)
    # print(_,y_pred)
    correct+=int(sum(y_pred==Y))
    wrong+=len(Y)-correct
    print("==============")
    print("Correct predictions: %d, Accuracy: %f" % (correct, correct / (correct + wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num=10
    batch_size=20
    train_sample_num=5000
    input_size=6
    learing_rate=0.05

    model=TorchModel(input_size)
    optim=torch.optim.Adam(model.parameters(),lr=learing_rate)
    log=[]

    train_X,train_Y=build_dataset(train_sample_num)

    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch_index in range(train_sample_num//batch_size):
            x=train_X[batch_index*batch_size:(batch_index+1)*batch_size]
            y=train_Y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss=model(x,y) #自动forward
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\nEpoch %d, Average Loss: %f" % (epoch + 1, np.mean(watch_loss)))
        # accuracy = evaluate(model)
        # print(accuracy)
        # log.append([accuracy, float(np.mean(watch_loss))])
        # print(log)
    torch.save(model.state_dict(),"model.pth")
    return

def predict(model_pth,input_vector):
    input_size=6
    model=TorchModel(input_size)
    model.load_state_dict(torch.load(model_pth))

    model.eval()
    with torch.no_grad():
        pre_result=model.forward(torch.FloatTensor(input_vector))
    for vector,result in zip(input_vector,pre_result):
        print("Input: %s, Predicted Class: %d, Probability: %f" % (vector, int(result.argmax()), result.max()))

if __name__ == '__main__':
    # 训练模型并保存
    main()
    
    test_vec = [
        [0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843, 0.18920843],
        [0.04963533, 0.9524256, 0.95758807, 0.9520434, 0.84890681, 0.18920843],
        [0.0797868, 0.007482528, 0.53625847, 0.34675372, 0.09871392, 0.28920843],
        [0.89349776, 0.59416669, 0.02579291, 0.41567412, 0.9358894, 0.98920843]
    ]
    predict("model.pth", test_vec)
