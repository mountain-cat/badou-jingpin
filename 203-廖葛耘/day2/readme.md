### 简介
随机10维向量，以固定式加法转换成5维向量后从中拿出最高值作为输出（真实值）

##### 模型初始化
```python 
def __init__(self, input_size):
    super(TorchModel, self).__init__()
    # 进行5层分类
    self.linear = nn.Linear(input_size, 5)
    # 使用Softmax和交叉熵解决多分类问题
    self.activation = nn.Softmax(dim=1)
    self.loss = nn.CrossEntropyLoss
``` 
这边由于是多分类问题所以会通过softmax作为激活函数，交叉熵作为损失函数去处理

##### 模型预测
```python 
x = self.linear(x)
y_pred =  self.linear(x)
if y is not None:
    return self.loss(y_pred, y)  # 预测值和真实值计算损失
else:
    return y_pred  # 输出预测结果
```
借用了transformer开发规范，如果有真实值的情况会返回损失函数；反之会返回模型预测结果

#####  样本生成方法
```python 
x = np.random.random(10)
res = [x[0]+x[3],x[5]+x[1],x[9]+x[2],x[7]+x[5],x[4]+x[4]]
return x, res.index(max(res))
``` 
确保数据是有规律的同时模型也不会直接了当的发现怎么处理


#####  配置参数
```python 
epoch_num = 40  # 训练轮数
batch_size = 20  # 每次训练样本个数
train_sample = 5000  # 每轮训练总共训练的样本总数
test_sample = 100 # 每轮的测试集数据
input_size = 10  # 输入向量维度
learning_rate = 0.001  # 学习率
```
这个是测试后估算出来最适合的训练参数

##### 模型训练结果
![image](https://github.com/koklinliau/badou-jingpin/assets/140817016/68c6ca4e-c926-4540-adde-ae1e8da5c425)

##### 模型最后几次训练的结果
![image](https://github.com/koklinliau/badou-jingpin/assets/140817016/969910a1-d005-4d45-8117-814f9093d76f)

