import jieba
from collections import defaultdict
import pandas as pd
import time



jieba.initialize()


class MyBayes :
    def __init__(self, path):
        
        self.word_freqs = defaultdict(dict)
        self.label_times = defaultdict(int)
        self.load(path)
        
        
    def load(self,path):
        df = pd.read_csv(path)
        result = defaultdict(dict)
        
        for _, value in df.iterrows():
            
            label_name = value['label']
            words = jieba.lcut(value['review'])
            # print(value['review'])
            # print( words)
            # print([w[0] for w in list(words)])
            self.label_times[label_name] += 1
            word_freq = self.word_freqs [label_name]
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else :word_freq[word ] += 1
            
            
            # time.sleep(2)
        self.freq_to_prob()
        # print(self.word_freqs[1])
        # print(self.label_times)
        return
    
    
    def freq_to_prob(self):
        total_label_times = sum(self.label_times.values())
        self.label_times = dict([c, self.label_times[c]/total_label_times ] 
                                for c in self.label_times)
        self.word_prob = defaultdict(dict)
        for classname, word_dict in self.word_freqs.items():
            # print(classname)
            # print(len(word_dict))
            total_word_times = sum(a for a in word_dict.values())
            # print(total_label_times)
            
            for word in word_dict:
                
                prob =( word_dict[word ]+1 )/total_word_times
                self.word_prob[classname][word] = prob
            self.word_prob[classname]['<unk>'] = 1/total_word_times
        
        # print(self.label_times)
        return
    def get_word_prob(self, words, classname):
        result = 1
        for word in words:
            unkprob = self.word_prob[classname]['<unk>']
            result *= self.word_prob[classname].get(word, unkprob)
        return result
    
    def get_class_prob(self, words, classname):
        p_x = self.label_times[classname]
        p_w_x = self.get_word_prob(words, classname)
        
        return p_x* p_w_x
    
    def classify(self, sentence):
        words = jieba.lcut(sentence)
        results = []
        for classname in self.label_times:
            prob = self.get_class_prob(words, classname)
            results.append(["好评" if classname == 1 else "差评", prob] )
        results = sorted (results, key = lambda x: x[1], reverse= True)
        
        for classname, prob in results:
            print("属于{}的概率{}".format(classname, prob))
        
        
        
        
        


if __name__ == "__main__":
    m = MyBayes('./文本分类练习.csv')
    query = '不好吃'
    m.classify(query)
        