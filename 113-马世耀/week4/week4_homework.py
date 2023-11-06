#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-08-10 10:48:12
LastEditors: Shiyao Ma
LastEditTime: 2023-08-11 10:51:24
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
#week4作业

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


def all_cut(sentence: str, Dict):
    """ Another method, from the beginning to the end.
    """
    dict_max_len = sorted([len(ele) for ele in Dict.keys()], key = lambda x: x, reverse = True)[0]

    if len(sentence) == 1 and sentence in Dict.keys():
        return [sentence]

    RESULT = []
    inprogress = [([], sentence)]
    
    while len(inprogress) > 0:
        cond, remains = inprogress[0]

        if remains == '':
            RESULT.append(cond)

        for i in range(1, dict_max_len + 1):
            if len(remains) < i:
                break
            cond_copy = cond.copy()
            if remains[:i] in Dict.keys():
                cond_copy.append(remains[:i])
                inprogress.append((cond_copy, remains[i:]))
            else:
                continue

        inprogress.pop(0)

    return RESULT


result = all_cut(sentence, Dict)
print(f"{result}\nIn total, {len(result)} possibilities as above")


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