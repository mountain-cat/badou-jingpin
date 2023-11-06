#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
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

class Node:
    def __init__(self,level,str,pidx):
        self.level=level
        self.str=str
        self.pidx=pidx

#通过递归调用，得出所有可能的切分方案
#用Dict组成一棵树，切分到最后一个字符作为叶子结点，然后回溯遍历组成字符串
#level 树的层级 pidx父节点的序号
def doCut(sentence,tree,leafList,level,pidx):
    s=len(sentence)
    nodeList=[]
    if level in tree:
        nodeList=tree[level]
    else:
        tree[level]=nodeList
    for i in range(1,s+1):
        str=sentence[0:i]
        if str in Dict:
            node=Node(level,str,pidx)
            nodeList.append(node)
            if i==s:
                leafList.append(node)
            else:
                doCut(sentence[i:s], tree, leafList,level+1, len(nodeList)-1)

def getResult(node,tree,substr):
    level=node.level
    pidx=node.pidx
    str=node.str
    if level>0:
        pnode=tree[level-1][pidx]
        result=getResult(pnode, tree, ","+str+substr)
    else:
        result=str+substr
    return result


sentence = "经常有意见分歧"
tree={}
leafList=[]
doCut(sentence,tree,leafList,0,-1)
for leafnode in leafList:
    print(getResult(leafnode,tree,""))




