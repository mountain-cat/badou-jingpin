import matplotlib.pyplot as pyplot
import math




X = [0.01*i for  i in  range(100)]
Y = [2 * x**2 + 3*x + 4 for x in X ]
pyplot.scatter(X, Y)
pyplot.show()
def func(x):
    y = w1* x**2 + w2 * x + w3
    return  y
#定义一个简单的模型，w1w2w2是权重

def loss(y_pred,y_true):
    return (y_pred - y_true)**2
#定义损失函数,原式=（w1 * x**2 + w2 * x + w3-y_true）**2

w1,w2,w3=2,3,5
lr=0.1#学习率
loss_list=[]#定义一个空列表存放loss



for epoch in range(3000):
    epoch_loss=0
    for x,y_true in zip(X,Y):
     y_pred = func(x)
    epoch_loss +=loss(y_pred,y_true)
    #将所有的LOSS进行累加

    #梯度计算
    grad_w1 = 2*(y_pred - y_true) * x**2
    grad_w2 = 2*(y_pred - y_true) * x
    grad_w3 = 2*(y_pred - y_true)
    #根据梯度计算出权重
    w1 = w1-lr * grad_w1
    w2 = w2-lr * grad_w2
    w3 = w3-lr * grad_w3

    epoch_loss=epoch_loss/len(X)
    #计算出平均loss
    loss_list.append(epoch_loss)
    if epoch_loss < 0.000001:
        break
    # print(f"第{epoch}轮， loss {epoch_loss}")
    print(loss_list)






