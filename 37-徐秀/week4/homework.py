# week4作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"


def cut_sentence(sentence_lst, dict, max_length):
    """切分句子方法"""
    cut_res_temp = list()
    for item in sentence_lst:
        for i in range(max_length):
            cut_str = item[-1][0:i + 1]
            if cut_str in dict:
                keep = item[:-1]
                keep.extend([item[-1][0:i + 1], item[-1][i + 1:]])
                if keep not in cut_res_temp and keep[-1]:
                    cut_res_temp.append(keep)
    # print(cut_res_temp)
    return cut_res_temp


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, dict):
    max_length = max([len(key) for key in dict.keys()])
    # 输出的最终结果
    target_result = []
    sentence_ls = [[sentence]]
    # 记录前一次切分结果
    cut_res_last = []
    while True:
        cut_res_ls = cut_sentence(sentence_ls, dict, max_length)
        cut_res_ls.sort(reverse=True)
        # 两次切分长度和结果一致，切分结束
        if cut_res_last == cut_res_ls:
            break
        cut_res_last = cut_res_ls
        for item in cut_res_ls:
            if item[-1] in dict and item not in target_result:
                target_result.append(item)
        # 更新最新的切分数据
        sentence_ls = cut_res_ls
    target_result.sort(reverse=True)
    for item in target_result:
        print(item)
    return target


# 目标输出;顺序不重要
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

if __name__ == '__main__':
    all_cut(sentence, dict)
