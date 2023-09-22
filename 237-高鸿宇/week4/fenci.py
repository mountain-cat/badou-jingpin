import jieba

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

def calc_dag(sentence):
    '''
    生成一个有向无环图。
    '''
    DAG = dict()
    n = len(sentence)         # 计算句子的长度
    for k in range(n):
        temp = []
        frag = sentence[k]
        i = k
        while i < n:
            if frag in Dict:
                temp.append(i)
            i += 1
            frag = sentence[k:i+1]
        if not temp:
            # 如果temp是空，说明词表中没有当前字,此字单独成一个token
            temp.append(k)
        DAG[k] = temp
    return DAG

sentence = "经常有意见分歧"
print(calc_dag(sentence))
#结果应该为{0: [0, 1], 1: [1], 2: [2, 4], 3: [3, 4], 4: [4, 6], 5: [5, 6], 6: [6]}

#将DAG中的信息解码（还原）出来，用文本展示出所有切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self, sentence):
        self.sentence = sentence
        self.length = len(sentence)
        self.dag = calc_dag(sentence)
        self.unfinished_path = [[]]
        self.finished_path = []

    def decode_next(self, path):
        n = len(''.join(path))
        if n == self.length:
            # 若当前长度与句子长度一致，说明当前情况的分词已经分完
            self.finished_path.append(path)
            return
        candidates = self.dag[n]
        temp = []
        for each in candidates:
            temp.append(path + [self.sentence[n:each+1]])
        self.unfinished_path += temp
        return

    #递归调用序列解码过程
    def decode(self):
        while self.unfinished_path != []:
            # 获取当前的token
            path = self.unfinished_path.pop()
            self.decode_next(path)


sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finished_path)