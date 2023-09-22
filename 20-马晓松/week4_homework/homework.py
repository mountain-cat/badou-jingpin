#week3作业

#模仿week4的答案来完成的answer
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

#这个代码对待切分文本进行了全面的切分，最后的返回结果存储了所有的切分信息
def calc_dag(sentence):
    DAG = {}
    N = len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < N:
            if frag in Dict:
                tmplist.append(i)
            i+=1
            frag = sentence[k : i+1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG

#将DAG中的所有信息进行还原，用文本展现书所有的切分方式
class DAGDecode:
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = calc_dag(sentence)
        self.length = len(sentence)
        self.unfinish_path = [[]]
        self.finish_path = []
#对于每一个序列，如果已经解码完毕，则直接添加到finish_path, 如果没有解码完毕，则放到待解码队列
    def decode_next(self, path):
        path_length = len("".join(path))
        print("1111111111")
        print(path_length)
        if path_length == self.length:
            self.finish_path.append(path)
            print("222222222")
            print(self.finish_path)
            return
        candidates = self.DAG[path_length]
        new_path=[]
        for candidate in candidates:
            new_path.append(path+[self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_path
        print("333333333333")
        print(self.unfinish_path)
        return
#递归调用序列解码
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()
            self.decode_next(path)

sentence = "经常有意见分歧"
print(calc_dag(sentence))
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)
# #目标输出;顺序不重要
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