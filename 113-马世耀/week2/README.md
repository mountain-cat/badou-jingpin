<!--
 * @Author: Shiyao Ma
 * @Date: 2023-07-28 11:14:22
 * @LastEditors: Shiyao Ma
 * @LastEditTime: 2023-07-28 17:22:25
 * Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
-->
# **目标：尝试将torchDemo脚本改造成一个多分类训练任务**

## 规律
* x是一个5维向量，一共3个类别
* 如果第1个数为最大则为类别0
* 如果第1个数为最小则为类别1
* 其余情况则为类别2
* 理论上，生成的标签中类别0:1:2 ~= 1:1:3，可设置balanced = True对生成样本的类别进行平衡

## 模型架构
* 单层linear层，50神经元，ReLU激活函数
* 接Linear&Softmax分类器
* loss选用CE loss