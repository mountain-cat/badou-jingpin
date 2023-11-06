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

# 通过有向无环图的逻辑去分割
def word_dag(sentence, word_max):
    sentence_index = dict()
    for i in range(len(sentence)):
        # 把首个字保存起来方便后做比较
        word = sentence[i]
        # 检查该字是否有在词典里边，有的话就创建索引
        if sentence[i] in Dict:
            sentence_index[i] = [i]
        # 根据要查找的长度操作
        for num in range(1, word_max):
            # 如果要查找的索引超出句子的长度则停止查找
            if num + i >= len(sentence):
                break
            # 凑起来
            word += sentence[i + num]
            # 如果词汇在词典里边则添加进单词索引里
            if word in Dict:
                if i in sentence_index:
                    sentence_index[i].append(i + num)
                else:
                    sentence_index[i] = [i + num]
    return sentence_index

class SentenceParticiple:
    def __init__(self, sentence, word_max):
        self.sentence = sentence
        self.DAG = word_dag(sentence, word_max)

    def execute(self, start=0):
        # 递归函数，生成所有可能的切分方式
        # start: 当前处理的字符索引
        if start >= len(self.sentence):
            # 如果当前索引超出句子长度，返回空切分
            return [[]]

        if start in self.DAG:
            # 如果当前索引在有向无环图中存在
            splits = []
            for end in self.DAG[start]:
                # 对于每个可能的结束索引，递归生成子切分
                sub_splits = self.execute(end + 1)
                for sub_split in sub_splits:
                    # 将当前切分添加到子切分的前面
                    splits.append([self.sentence[start:end + 1]] + sub_split)
            return splits
        else:
            # 如果当前索引不在有向无环图中，继续下一个字符
            return self.execute(start + 1)

# 待切分文本
sentence = "经常有意见分歧"
word_max = 5
DAG = SentenceParticiple(sentence, word_max)
result = DAG.execute()

# 目标输出
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

# # 检查结果是否与目标输出一致
# assert result == target, "Results do not match target"
#
# # 打印结果
for r in result:
    print(r)
