def full_segment(text, dictionary):
    """
    对文本进行全切分
    """
    result = [] # 保存切分结果
    length = len(text)

    def dfs(start, segments):
        if start == length: # 已经处理到文本末端
            result.append(segments)
            return
        for i in range(start+1, length+1):
            segment = text[start:i]  # 获取当前分段
            if segment in dictionary: # 如果存在于词典中，则可以继续分割
                dfs(i, segments + [segment])

    dfs(0, [])
    return result

text = "经常有意见分歧"
dictionary = {  "经常":0.1,
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
segments = full_segment(text, dictionary)  # 获取所有可能的切分

print("全切分结果：")
for seg in segments:
    print("/".join(seg))  # 输出所有可能的切分，使用斜杠连接各个词语


# def dfs(start, path):
#     if start == -1:
#         # 如果已经到了切分点的末尾，则找到一组切分方案
#         print(path[::-1])
#         return
#     for i in range(start + 1):
#         seg = txt[i:start + 1]
#         if seg in word_dict:
#             # 如果当前子串是个单词，则尝试继续从当前位置前面的位置开始切分
#             dfs(i - 1, [seg] + path)
#     return
#
#
# # 使用示例
# txt = "经常有意见分歧"
# word_dict = Dict = {"经常":0.1,
#         "经":0.05,
#         "有":0.1,
#         "常":0.001,
#         "有意见":0.1,
#         "歧":0.001,
#         "意见":0.2,
#         "分歧":0.2,
#         "见":0.05,
#         "意":0.05,
#         "见分歧":0.05,
#         "分":0.1}
# start_pos = len(txt) - 1
# dfs(start_pos, [])  # 从后往前切分
