#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
import copy
from collections import defaultdict

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

word_string=[]


##测试版本：
def sentence_cut(sentence,word_string):
    if len(sentence)==0:
        print(word_string)
        return
    words = [dict_word for dict_word in Dict if sentence.startswith(dict_word)]
    #print(words)
    #input()
    for index, word in enumerate(words):
        word_str=copy.copy(word_string)
        #print(word)
        sentencee=sentence
        sentencee=sentencee[len(word):]
        #print(sentencee)
        #word_str.append(word_string)
        word_str.append(word)
        sentence_cut(sentencee,word_str)

if __name__ == "__main__":
    sentence_cut(sentence,word_string)
# ['经常', '有', '意见', '分歧']
# ['经常', '有', '意见', '分', '歧']
# ['经常', '有', '意', '见', '分歧']
# ['经常', '有', '意', '见', '分', '歧']
# ['经常', '有', '意', '见分歧']
# ['经常', '有意见', '分歧']
# ['经常', '有意见', '分', '歧']
# ['经', '常', '有', '意见', '分歧']
# ['经', '常', '有', '意见', '分', '歧']
# ['经', '常', '有', '意', '见', '分歧']
# ['经', '常', '有', '意', '见', '分', '歧']
# ['经', '常', '有', '意', '见分歧']
# ['经', '常', '有意见', '分歧']
# ['经', '常', '有意见', '分', '歧']


##############################################################################################################
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    return target

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
