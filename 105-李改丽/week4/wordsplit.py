def load_pre_word_dict(path):
    max_word_length = 0
    word_dict = {}  # 用set也是可以的。用list会很慢
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 1

            word = word[:len(word)-1]
            while word != "":
                if word not in word_dict:
                    word_dict[word] = 0
                word = word[:len(word) - 1]

            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length

def load_prefix_wordlist(path):
    word_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.strip()[0]
            for i in range(1, len(word)):
                if word[:i] not in word_dict:
                    word_dict[word[:i]] = 0  #前缀

            word_dict[word] = 1  #词
    return word_dict

def cut_method2_pre2(string, word_dict):
    word_list = []
    start_index = 0
    end_index = 1
    word_find = ""
    word = string[start_index:end_index]
    # print(word)
    word_find = word
    while start_index < len(string) and end_index < len(string):
        #word = string[start_index:end_index]
        if word in word_dict:
            if word_dict[word] == 1:
                word_find = word
                end_index += 1           #find word
                word = string[start_index:end_index]
                # print("1---------- ", start_index, end_index)
                # print(word)
            else:
                # print("0---------- ", start_index)
                end_index += 1           # find prefix
                word = string[start_index:end_index]
                # print(end_index)
                # print(word)
        else:
            word_list.append(word_find)
            start_index += len(word_find)
            end_index = start_index + 1
            word = string[start_index:end_index]
            word_find = word
            # print(word)

    word = string[start_index:]
    # print(word)
    word_list.append(word)
    return word_list

#输入字符串和字典，返回词的列表
def cut_method2_pre3(string, prefix_dict):
    if string == "":
        return []
    words = []  # 准备用于放入切好的词
    start_index, end_index = 0, 1  #记录窗口的起始位置
    window = string[start_index:end_index] #从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index < len(string):
        #窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            # print("1: ", window)
            # print(find_word)
            words.append(find_word)  #记录找到的词
            # print(words)
            start_index += len(find_word)  #更新起点的位置
            end_index = start_index + 1
            window = string[start_index:end_index]  #从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词
        elif prefix_dict[window] == 1:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = string[start_index:end_index]
        #窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    #最后找到的window如果不在词典里，把单独的字加入切词结果
    # if prefix_dict.get(window) != 1:
    #     print("  -----2----", window)
    #     words += list(window)
    #     print("in get: ", words)
    # else:
    #     words.append(window)
    return words
def cut_method1_pre(string, word_dict_pre):
    words = []
    while string != '':
        n = 1
        word = string[:n]
        while word in word_dict_pre:
            if word_dict_pre[word] == 1:
                words.append(word)
                string = string[len(word):]
                word = ""
                break
            else:
                n += 1
                if n > len(string):
                    break
                else:
                    word = string[:n]

        if word == "":
            continue
        else:
            word = word[:1]
            words.append(word)
            string = string[len(word):]

    return words

def get_wordlist(path):
    word_dict = {}
    max_word_length = 0
    with open(path, encoding = "utf8") as f:
        for line in f:
            # print(line.split())
            # print(type(line.split()))
            # print(line.split()[0])
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length


#正向最大匹配
def cut_method1(string, word_dict, max_len):
    word_result = []
    while string != "":
        lens = min(max_len, len(string))
        word = string[:lens]
        #print(word)
        while word not in word_dict:
            if len(word) == 1:
                break
            else:
                word = word[:len(word)-1]
        word_result.append(word)
        string = string[len(word):]
    return word_result

#最大反向匹配
def cut_method2(string, word_dict, max_len):
    word_result = []
    while string != "":
        lens = min(len(string), max_len)
        word = string[len(string)-lens:]
        while word not in word_dict:
            if len(word) == 1:
                break
            else:
                word = word[1:]
        word_result.append(word)
        string = string[:len(string) - len(word)]
    word_result_forward = []
    for i in range(len(word_result)):
        word_result_forward.append(word_result[len(word_result)-1 - i])
    return word_result_forward

def compare_word(wordlist1, wordlist2):
    pass


path = "词表.txt"
sentence = "北京大学生前来报到吃"

def main1(path):
    word_dict, max_word_length = get_wordlist(path)
    #print("词表： ", word_dict)
    #print("max_word_length: ", max_word_length)
    print(len(sentence) - 2)
    word_result1 = cut_method1(sentence, word_dict, max_word_length)
    print(word_result1)
    word_result2 = cut_method2(sentence, word_dict, max_word_length)
    print(word_result2)
    return

def main2(path):
    word_dict, max_len = load_pre_word_dict(path)
    #print("词表： ", word_dict)
    word_result1 = cut_method1_pre(sentence, word_dict)
    print(word_result1)
    word_result2 = cut_method2_pre2(sentence, word_dict)
    print(word_result2)
    word_result2 = cut_method2_pre3(sentence, word_dict)
    print(word_result2)
    return
main1(path)
main2(path)