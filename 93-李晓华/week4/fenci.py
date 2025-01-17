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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 初始化结果列表
    result = []

    # 定义递归函数，用于遍历所有可能的切分组合
    def dfs(start, path):
        # 当遍历到句子末尾时，将当前路径添加到结果列表中
        if start == len(sentence):
            result.append(path[:])
            return

        # 遍历当前位置到句子末尾的所有子串
        for end in range(start + 1, len(sentence) + 1):
            # 如果子串在词典中，则将子串添加到当前路径，并继续递归遍历
            if sentence[start:end] in Dict:
                path.append(sentence[start:end])
                dfs(end, path)
                path.pop()  #`path.pop()` 这部分代码的作用是在递归遍历过程中，回溯到上一层时，将当前路径（`path`）中的最后一个元素（即当前子串）移除。这样做的目的是为了在回溯时，恢复到上一层的状态，以便继续尝试其他可能的切分组合。\n\n在 `dfs` 函数中，我们使用深度优先搜索（DFS）的方法遍历所有可能的切分组合。当我们找到一个在词典中的子串时，我们将其添加到当前路径（`path`）中，并继续递归遍历。当递归返回时，表示当前子串的后续部分已经遍历完成，我们需要回溯到上一层，尝试其他可能的子串。因此，我们需要将当前子串从路径（`path`）中移除，恢复到上一层的状态。这就是 `path.pop()` 的作用。

    # 从句子开头开始递归遍历
    dfs(0, [])

    return result

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

# 调用全切分函数，输出根据字典能够切分出的所有的切分方式
target = all_cut(sentence, Dict)

# 打印结果
print(target)