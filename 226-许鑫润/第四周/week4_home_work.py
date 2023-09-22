import jieba
#week3作业
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式

def all_cut(sentence, Dict):
    target = []   #最终输出
    ret_list = [] #每一轮的输出
    level = 0     #当前位于第几层
    __all_cut(sentence, Dict, target, level, ret_list) #没有什么问题是封装无法解决的，如果有就多封装一层
    target = sorted(target, reverse=True)
    return target

#__all_cut定义为当给定句子为sentence时，第level层从第一个字开始的词汇选择情况
def __all_cut(sentence, Dict, target, level, ret_list):

    #边界条件 最后一层再将这一轮的输出放入
    if len(sentence) == 0:
        target.append(ret_list)
        return
    elif len(sentence) == 1: #只剩一个字时只有一种切分方式
        ret_list.append(sentence)
        target.append(ret_list)
        return

    for e in range(len(sentence)): #当前层词汇是从第一个字开始的只需要遍历结束位置 + 1
        e += 1

        stemp = sentence[0:e]
        if stemp not in Dict: #当前层的词汇不符合要求应该跳过
            continue

        if level == 0: #第一层清空所有
            ret_list = []

        ret_list.append(stemp) #当前层词汇的选取情况

        sentence_next = sentence[e:] #下一层句子

        level += 1 #进入下一层

        __all_cut(sentence_next, Dict, target, level, ret_list) #递归阶段，问题的规模通过缩短句子缩小了

        #回溯阶段撤销之前的修改，包括当前层之后所有层选择的情况
        level -= 1
        ret_list = ret_list[0:level]

    return

ret = all_cut(sentence, Dict)

for i, r in enumerate(ret):
    print("r[%d]:%s" % (i + 1, r))

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]