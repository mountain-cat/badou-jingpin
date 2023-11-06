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

# 切分结果
split_result = []


def all_cut(sentence, dict,link_list=[]):

    if len(sentence) <= 0:
        split_result.append(link_list.copy())
        return

    print(f"sentence:{sentence}")

    # 获取首词
    head_char = sentence[:1]
    # 用首词去查词典
    keys = list(filter(lambda k: k[:len(head_char)] == head_char, dict))

    # 遍历词典
    for key in keys:
        splist_list = link_list.copy()
        splist_list.append(key)
        print(f" --- next sentence:{sentence[len(key):]},next head:{key}")
        all_cut(sentence[len(key):],dict, link_list=splist_list)


if __name__ == "__main__":
    all_cut(sentence,list(Dict.keys()))
    print("\n========== 结果输出 ==========\n")
    print(*split_result, sep="\n")


#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]