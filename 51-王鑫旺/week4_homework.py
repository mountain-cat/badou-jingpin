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

def all_cut(s, word_dict):
    n = len(s)
    dp = [False] * (n + 1)  # 动态规划数组，初始化为 False
    dp[0] = True  # 空串默认为可切分

    word_break_points = [[] for _ in range(n + 1)]  # 记录每个位置可能的切分点

    for i in range(1, n + 1):
        # print(dp)
        for j in range(i):
            # 检查从位置 0 到位置 j-1 的子串是否可切分，并且位置 j 到 i-1 的子串是否在字典中存在
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True  # 更新动态规划数组中的值为 True
                word_break_points[i].append(j)  # 记录切分点 j
    
    return generate_sentences(s, word_break_points, n)

def generate_sentences(s, word_break_points, end):
    if end == 0:
        return []

    sentences = []
    for start in word_break_points[end]:
        word = s[start:end]
       
        if start == 0:
            sentences.append(word)  # 在位置 0 处，直接添加单词
           
        else:
            for prev_sentence in generate_sentences(s, word_break_points, start):
                # print(prev_sentence)
                sentences.append(prev_sentence + " " + word)  # 递归生成句子
    
   

    return sentences

result=all_cut(sentence, Dict)

target1=[]
for s in result:
    element=[]
    for word in s.split():
        element.append(word)
    target1.append(element)
        
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
#target1是最终切分的结果
print("结果和目标输出元素是否相同（不分顺序）：",sorted(target1)==sorted(target))