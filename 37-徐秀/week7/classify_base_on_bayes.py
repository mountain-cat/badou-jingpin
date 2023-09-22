import jieba
import json
from collections import defaultdict


class BayesApproach:
    def __init__(self, data_path):
        self.p_class = defaultdict(int)
        self.word_class_prob = defaultdict(dict)
        # 加载语料并保存获取的统计参数使用下面的方法
        self.load(data_path)
        self.save()
        # 直接加载保存好的统计结果使用self.load_result()
        # self.load_result()

    def load_result(self):
        with open('word_class_prob.json', mode='r', encoding='utf8') as f:
            self.word_class_prob = json.loads((f.read()))
        with open('p_class.json', mode='r', encoding='utf8') as f2:
            self.p_class = json.loads((f2.read()))

    def load(self, path):
        self.class_name_to_word_freq = defaultdict(dict)
        # 汇总一个词表
        self.all_words = set()
        with open(path, encoding='utf8') as f:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                else:
                    class_name, review = line.strip().split(',', 1)
                    words = jieba.lcut(review)
                    self.all_words.union(set(words))
                    # 记录每个类别样本数量
                    self.p_class[class_name] += 1
                    word_freq = self.class_name_to_word_freq[class_name]
                    # 记录每个类别下的词频
                    for word in words:
                        if word not in word_freq:
                            word_freq[word] = 1
                        else:
                            word_freq[word] += 1
        self.freq_to_prob()

    def freq_to_prob(self):
        """将记录的词频和样本频率都转化为概率"""
        total_sample_count = sum(self.p_class.values())
        self.p_class = dict([c, self.p_class[c] / total_sample_count] for c in self.p_class)
        # 词概率计算
        self.word_class_prob = defaultdict(dict)
        for class_name, word_freq in self.class_name_to_word_freq.items():
            # 每个类别总词数
            total_word_count = sum(count for count in word_freq.values())
            for word in word_freq:
                # 加1平滑，避免出现概率0，计算怕（wn|x1)
                prob = (word_freq[word] + 1) / (total_word_count + len(self.all_words))
                self.word_class_prob[class_name][word] = prob
            self.word_class_prob[class_name]['<unk>'] = 1 / (total_word_count + len(self.all_words))
        return

    def save(self):
        """保存根据语料获取到的统计结果"""
        with open('word_class_prob.json', mode='w', encoding='utf8') as f:
            # python的json.dumps默认使用的ASCII，使用中文时需要加ensure_ascii=False,这样可以使中文字符串按原样输出
            f.write(json.dumps(self.word_class_prob, ensure_ascii=False))
        with open('p_class.json', mode='w', encoding='utf8') as f:
            f.write(json.dumps(self.p_class, ensure_ascii=False))

    def get_words_class_prob(self, words, class_name):
        """ P(W1,W2,...Wn|X1) = P(W1|X1)*P(W2|X1)...P(Wn|X1) """
        result = 1
        for word in words:
            unk_prob = self.word_class_prob[class_name]['<unk>']
            result *= self.word_class_prob[class_name].get(word, unk_prob)
        return result

    def get_class_prob(self, words, class_name):
        """计算P(W1,W2,...Wn|X1)*P(X1)"""
        # 获取p(X1)
        p_x = self.p_class[class_name]
        # P(W1,W2,...Wn|X1) = P(W1|X1)*P(W2|X1)...P(Wn|X1)
        p_w_x = self.get_words_class_prob(words, class_name)
        return p_x * p_w_x

    # 做文本分类
    def classify(self, sentence):

        words = jieba.lcut(sentence)
        results = []
        for class_name in self.p_class:
            # 计算class_name类概率
            prob = self.get_class_prob(words, class_name)
            results.append([class_name, prob])
        # 排序
        results = sorted(results, key=lambda x: x[1], reverse=True)
        # 计算公共分母：P(W1,W2...Wn)=P(w1,w2,wn|x1)*p(x1)+P(w1,w2,wn|x2)*p(x2)
        # 不做这一步对排序也没影响，只不过得到的不是0-1之间的值
        pw = sum(x[1] for x in results)
        results = [[c, prob / pw] for c, prob in results]
        # 打印结果
        for class_name, prob in results:
            print("属于类别[%s]的概率为%f" % (class_name, prob))
        return results


if __name__ == '__main__':
    path = './文本分类练习.csv'
    ba = BayesApproach(path)
    query = "火腿鸡蛋饼，看到鸡蛋了"
    ba.classify(query)
    """
    属于类别[0]的概率为0.982480
    属于类别[1]的概率为0.017520
    """
