# 词典
Dict = {"经常": 0.1,
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


def calc_dag(sentence):
    DAG = {}
    N = len(sentence)
    for i in range(N):
        k = 1
        DAG[i] = [i]
        while i + k < N:
            if sentence[i:i + k + 1] in Dict:
                DAG[i].append(i + k)
            k = k + 1
    return DAG


class DAGDecode:
    def __init__(self, sentence):
        self.sentence = sentence
        self.len_sen = len(sentence)
        self.DAG = calc_dag(sentence)
        self.unfinish = [[]]
        self.finish = []
        self.decode()

    def decode(self):
        while self.unfinish != []:
            unfinish_path = self.unfinish.pop()
            self.unfinish_decode(unfinish_path)

    def unfinish_decode(self, path):
        unfinish_path = "".join(path)
        unfinish_len = len(unfinish_path)
        if unfinish_len == self.len_sen:
            self.finish.append(path)
            return
        points = self.DAG[unfinish_len]
        for point in points:
            little_path = [self.sentence[unfinish_len:point + 1]]
            self.unfinish.append(path + little_path)


sentence = "经常有意见分歧"
print(calc_dag(sentence))
print(DAGDecode(sentence).finish)
