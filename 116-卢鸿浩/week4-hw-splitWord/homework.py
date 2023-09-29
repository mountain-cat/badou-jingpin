#week3作业

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
#sentence = "经常有意见分歧"
sentence = "经常有意经经经常有意见分歧经常经常有意见分歧有意见分歧经经常有意见分经常有意经经经常有意见"

class StringTree:
    def __init__(self, strLeft, strRight, deep):
        self.strLeft = strLeft           # 不再划分的子字符串 --从左到右结合
        self.strRight = strRight         # 可继续划分的子字符串
        self.isOutNode = False           # 标识当前节点可以作为输出节点
        self.deep = deep + 1             # 当前的深度
        self.subTreeNum = 0              # 子树的个数
        self.subTree = []

    # 递归地把当前树划分为子树
    def splitToSubtree(self, strDict, maxKeyLen):
        # 判断当前节点可以作为输出节点
        if strDict.get(self.strRight) is not None:
            self.isOutNode = True
        else:
            self.isOutNode = False

        # 根据字典，递归地把右子字符串进行划分，把子树存储在subTree中
        for i in range(len(self.strRight)-1):
            if i < maxKeyLen and ((strDict.get(self.strRight[:i+1])) is not None):
                oneSubTree = StringTree(self.strRight[:i+1], self.strRight[i+1:], self.deep)
                oneSubTree.splitToSubtree(strDict, maxKeyLen)
                self.subTree.append(oneSubTree)
                self.subTreeNum += 1

    # 计算字典中key的最大字符串长度
    def getMaxLenOfKey(self, strDict):
        maxLenOfKey = 0
        for key in strDict:
            maxLenOfKey = maxLenOfKey if maxLenOfKey > len(key) else len(key)
        print("maxlenofkey:", maxLenOfKey)
        return maxLenOfKey

    # 后续遍历树，输出划分结果
    def postOrderTransversal(self):
        outArrays = []
        if self.isOutNode is True:
            outArrays.append([self.strLeft, self.strRight])
        if self.subTreeNum > 0:
            for subtreeindex in range(self.subTreeNum):  # 遍历每一个子树
                subOutArrays = self.subTree[subtreeindex].postOrderTransversal()
                for i in range(len(subOutArrays)):
                    if self.deep != 1:
                        subOutArrays[i].insert(0, self.strLeft)  # 添加左子字符串到每个子树得到的字符串数组
                    outArrays.append(subOutArrays[i])
        return outArrays



#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    print(Dict.get('分'))
    print(Dict['分'])
    #TODO
    strTree = StringTree('', sentence, 0)
    strTree.splitToSubtree(Dict, strTree.getMaxLenOfKey(Dict))
    target = strTree.postOrderTransversal()
    for i in range(len(target)):
        print(target[i])
    print("num of cases:",len(target))
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

def main():
    all_cut(sentence, Dict)


if __name__=="__main__":
    main()