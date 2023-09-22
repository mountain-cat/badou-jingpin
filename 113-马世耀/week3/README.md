<!--
 * @Author: Shiyao Ma
 * @Date: 2023-07-28 11:14:22
 * @LastEditors: Shiyao Ma
 * @LastEditTime: 2023-08-03 16:49:29
 * Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
-->
# **[WEEK3] 目标：基于nlp的多分类任务示例，使用了rnn代替池化层**

## 规律
* x是一个字符串，包含6个字符
* 设定TARGET_CHARS为'a', 'b', 'c'三个字符
* 如果3个字符中有1个在x中，则类别为1，有2个出现则为类别2，有3个出现则为类别3，没有出现则为类别0
* build_dataset()方法可设置balanced = True对生成样本的类别进行平衡

## 模型架构
* pre层取embedding
* 单层RNN
* 线性分类器，4类
* loss选用CE loss