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
c=[]
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def get_DAG(sentence):
    DAG={}
    n=len(sentence)
    for k in range(n):
        l=[]
        for i in range(k,n):
            if sentence[k:i+1] in Dict:
                l.append(i)
        DAG[k]=l
    return DAG


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


#将DAG中的信息解码（还原）出来，用文本展示出所有切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = get_DAG(sentence)  #使用了上方的函数
        self.length = len(sentence)
        self.unfinish_path = [[]]   #保存待解码序列的队列
        self.finish_path = []  #保存解码完成的序列的队列

    #对于每一个序列，检查是否需要继续解码
    #不需要继续解码的，放入解码完成队列
    #需要继续解码的，将生成的新队列，放入待解码队列
    #path形如:["经常", "有", "意见"]
    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:  #已完成解码
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_paths  #放入待解码对列
        return

    #递归调用序列解码过程
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop() #从待解码队列中取出一个序列
            self.decode_next(path)     #使用该序列进行解码


dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)
