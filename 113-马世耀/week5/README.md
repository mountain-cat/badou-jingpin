<!--
 * @Author: Shiyao Ma
 * @Date: 2023-07-28 11:14:22
 * @LastEditors: Shiyao Ma
 * @LastEditTime: 2023-08-17 17:55:44
 * Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
-->
# **[WEEK5] 目标：在基于kmeans的聚类中，增加类内相似度的计算
* 113-马世耀 week5 homework

## 设计
* 方法：新建一个dict：sentence_distance, 其中key为簇号，元素为一个list
* 对组内所有点与center进行欧氏距离计算，并在相应簇号的值上append
* 对所有簇的值求平均，即得到每个簇的类内平均距离
* 按照距离从低到高(相似度从高到低)排序打印出结果