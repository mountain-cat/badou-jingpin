<!--
 * @Author: Shiyao Ma
 * @Date: 2023-07-28 11:14:22
 * @LastEditors: Shiyao Ma
 * @LastEditTime: 2023-08-11 10:50:11
 * Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
-->
# **[WEEK4] 目标：根据词表切分出一句话的所有可能组合

## 设计
* 输入：一个字符串 str，一个词表 dict[str, float], key为词，value为词频（可忽略）
* 思路：
    * 从字符串第一个字符开始循环，只要剩余不为空，则继续循环
    * 每次循环，将剩余部分第一个字符开始，再循环选出所有可能的包含第一个字的切分方法，然后置于原列表中
    * 在内循环中，如果剩余部分长度小于当前可能词的token长度，则break
    * 如果剩余为空字符串，则将结果加入RESULT变量中
    * 返回RESULT即为该词所有可能的切分方式