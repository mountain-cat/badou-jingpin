# 将二分类代码改写成多分类任务    
沿用老师上课构造数据的规律，将代码改写成三分类任务。   
  
## 任务描述  
三分类任务：第一个数大于第五个数标签为0、第一个数小于第五个数标签为0、第一个数等于第五个数标签为0。  
  
## 代码  
1、采用softmax为三分类模型激活函数，并设置参数dim=1在行数据上进行运算；采用交叉熵作为三分类模型的损失函数。  
          self.activation = nn.Softmax(dim=1)  # 使用Softmax作为激活函数  
          self.loss = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数  
            
2、在build_sample函数中创建符合任务要求的数据  
          if x[0] > x[4]:  
            return x, 0  # 第1个数大于第5个数，标记为0（正样本）  
          elif x[0] < x[4]:  
            return x, 1  # 第1个数小于第5个数，标记为1（负样本）  
          else:  
            return x, 2  # 第1个数等于第5个数，标记为2（中性样本）   
              
3、在evaluate函数中通过torch.argmax()查找概率值最大的值的索引表示类别  
          y_pred = torch.argmax(y_pred, dim=1)  # 找到概率最大的类别索引  
  
## 训练过程中损失值和结果    
![image](https://github.com/JhxCUGBCS/badou-jingpin/blob/main/91-%E5%A7%9C%E9%B9%A4%E7%A5%A5/week2/%E4%B8%89%E5%88%86%E7%B1%BB%E6%8D%9F%E5%A4%B1%E5%80%BC%E5%92%8C%E7%BB%93%E6%9E%9C.png)
  
## 预测结果演示  
![image](https://github.com/JhxCUGBCS/badou-jingpin/blob/main/91-%E5%A7%9C%E9%B9%A4%E7%A5%A5/week2/%E8%BF%90%E8%A1%8C%E7%BB%93%E6%9E%9C.jpg)
