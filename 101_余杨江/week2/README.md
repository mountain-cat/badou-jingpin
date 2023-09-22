# 作业记录
## 第二周
### 自己尝试修改 *TorchDemo.py* 为多分类，改写训练代码
- 二分类(0,1)激活函数为`sigmoid` 我改为三分类(-1,0,1)后使用`tanh`, loss 函数仍然使用`mse`?
- 参照 57 同学后，激活函数为 `softmax`。loss需要在 `main` 函数中定义(?)，多分类使用 `CrossEntropy`。
- 且参照 57 同学把，模型结构封装在`nn.Sequential()`中
- 
