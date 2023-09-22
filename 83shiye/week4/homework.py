#week3作业

def bfs(sentence, word_dict):
    one_step = []
    for window_size in range(len(sentence)+1):
        word = sentence[:window_size]
        if word in word_dict:
            one_step.append(word)
    return one_step


def use_dfs(sentence, word_dict):
    res = []
    path = []

    def dfs(sentence, word_dict):
        if len(sentence) == 0:
            res.append(path[:])
            return

        for window_size in range(len(sentence) + 1):
            word = sentence[:window_size]
            if word in word_dict:
                path.append(word)
                dfs(sentence[window_size:], word_dict)
                path.pop()

    dfs(sentence, word_dict)
    return res


def use_bfs(sentence, word_dict):
    res = []
    old = [[]]
    flag = True
    while flag:
        new = []
        count = 0
        for sen in old:
            assert isinstance(sen, list)
            length = len(''.join(sen))
            if length == len(sentence):
                new.extend([sen])
                count += 1
                if count == len(old):
                    res = old
                    flag = False
                    break
                continue
            one_step = bfs(sentence[length:], word_dict)
            if one_step:
                new.extend([sen + [x] for x in one_step])
        old = new
    return res


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
def all_cut(sentence, Dict, method=use_bfs):
    '''
    :param sentence: 句子
    :param Dict: 词典
    :param method: use_dfs or use_bfs
    :return:
    '''
    #TODO
    target = method(sentence, Dict)
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

if __name__ == "__main__":
    target_bfs = all_cut(sentence, Dict, use_bfs)
    target_dfs = all_cut(sentence, Dict, use_dfs)

    print('BFS result: \n', target_bfs)
    print('DFS result: \n', target_dfs)

