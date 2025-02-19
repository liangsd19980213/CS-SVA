import re
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import math

cpp_keywords = ['asm', 'auto', 'bool', 'break', 'case', 'catch', 'char', 'class', 'const', 'const_cast', 'continue',
                'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern',
                'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new',
                'operator', 'private', 'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short', 'signed',
                'sizeof', 'static', 'static_cast', 'struct', 'switch', 'template', 'this', 'throw', 'true', 'try',
                'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t',
                'while']
OUTPUT_DIR = '../model/codet5/output'


def camel_case_split(str):
    RE_CPP_WORDS = re.compile(r'''
    [a-zA-Z]+(?=[A-Z]) |     # 匹配驼峰命名中，紧跟大写字母的部分
    [a-z]+ |                # 匹配全小写单词
    [A-Z]+(?=[^a-zA-Z]|$) |  # 匹配全大写单词
    \d+ |                   # 匹配数字
    _+ |                    # 匹配下划线
    [^\w]+                  # 匹配非字母数字的符号
    ''', re.VERBOSE)
    return RE_CPP_WORDS.findall(str)


class StatementAnalyse():
    def __init__(self):
        self.output_dir = OUTPUT_DIR + '/attentions'
        self.tokenMap = {}
        self.globalTokenAttention = {}



    def statement_reader(self, statement_index, layer_num):
        """
        读取指定索引和层号的语句 Attention 信息，修复 tensor 数据格式问题。
        """
        statements = []
        path = os.path.join(self.output_dir, f"{statement_index}/layer_{layer_num}")
        with open(path, 'r') as f:
            while True:
                item = f.readline().strip()
                if not item:
                    break

                # 替换 tensor() 格式为浮点数
                item = item.replace('tensor(', '').replace(', device=\'cuda:0\')', '')

                # 清洗并跳过不完整或无效数据
                try:
                    statement = eval(item)  # 将字符串转换为列表
                    # 检查是否为全零
                    if all(val == 0.0 for val in statement):
                        continue
                    statements.append([float(val) for val in statement])
                except Exception as e:
                    print(f"Error parsing item: {item}, error: {e}")
                    continue
        return statements

    def read_global_token_attention(self, token):
        filename = OUTPUT_DIR + '/token_attentions/token_attentions'
        if len(self.globalTokenAttention) == 0:
            with open(filename, 'r') as f:
                item = 'blank_item'
                while True:
                    item = f.readline().strip()  # 去除空白字符
                    if not item:
                        break
                    key = item.split('  ||  ')[0]
                    value = float(item.split('  ||  ')[1])
                    self.globalTokenAttention[key] = value
        if token not in self.globalTokenAttention:
            # print(f"Warning: Token '{token}' not found in globalTokenAttention.")
            return 0.0
        return self.globalTokenAttention[token]


    def token_frequence_attention_map(self, tokens, statement_index, max_layer_num=12):
        statement_attentions = []
        statement_attentions = [self.statement_reader(statement_index, i)
                                for i in range(max_layer_num-1, max_layer_num)]
        average_attention = np.mean(np.array(statement_attentions), axis=0)
        for index, statement in enumerate(average_attention):
            need_calculate = True
            i = 0
            start = 0
            end = 0
            while i < len(statement):
                if statement[i] == 0.0:
                    if not need_calculate:
                        break
                    i = i + 1
                    continue
                else:
                    need_calculate = False

                    start = i
                    while i < len(statement):
                        if statement[i] == 0.0:
                            end = i
                            break
                        if tokens[i+1].startswith('Ġ'):
                            end = i + 1
                            break
                        else:
                            i = i + 1

                    token = ''.join(tokens[start:end])
                    camelCaseWord = []
                    # separate camel case
                    if token.startswith('Ġ'):
                        if len(token) > 1:
                            camelCaseWord = camel_case_split(token[1:])
                    else:
                        camelCaseWord = camel_case_split(token)
                    for token in camelCaseWord:
                        if token in self.tokenMap.keys():
                            frequence = self.tokenMap[token]['frequence']
                            attention = self.tokenMap[token]['attention']
                            self.tokenMap[token]['attention'] = \
                                (attention * frequence +
                                 sum(average_attention[index][start:end])/(end-start)) / (frequence + 1)
                            self.tokenMap[token]['frequence'] = frequence + 1
                        else:
                            self.tokenMap[token] = {'frequence': 1, 'attention': sum(
                                average_attention[index][start:end])/(end-start)}
                    start = end
                    i = i + 1


    def get_statement_attention(self, tokens, statement_index, in_statement_modulu=0.7, max_layer_num=12,
                                single_layer=False, layer_num=11):
        statement_attentions = []
        if not single_layer:
            statement_attentions = [self.statement_reader(statement_index, i) for i in range(max_layer_num-1, max_layer_num)]
        else:
            statement_attentions = [self.statement_reader(statement_index, layer_num)]
        average_attention = np.mean(np.array(statement_attentions), axis=0) * in_statement_modulu
        statement_attention = []
        for index, statement in enumerate(average_attention):
            need_calculate = True

            # calculate the average attention of the whole statement
            attention_sum = 0.0
            token_num = 0

            token_list = []

            for i in range(0, len(statement)):
                if statement[i] == 0.0:
                    if not need_calculate:
                        break
                    continue
                else:
                    # calculate the tokens' attention with the ratio of the whole dictionary and the attention in the
                    # statement

                    if tokens[i].startswith('Ġ'):
                        tokens[i] = tokens[i][1:]
                    # print(tokens[i])
                    global_token_attention_map = self.read_global_token_attention(tokens[i])
                    # print(global_token_attention_map)
                    token_attention = global_token_attention_map * (1-in_statement_modulu)
                    average_attention[index][i] += token_attention
                    token_list.append(tokens[i])
                    attention_sum += average_attention[index][i]
                    token_num += 1
                    need_calculate = False
            statement_attention.append({"token": token_list, "attention": (attention_sum / token_num)})
            # print(f"average_attention: {average_attention}")
            # print(f"statement_attention: {statement_attention}")
        return average_attention, statement_attention

    def output_statement(self, tokens, statement_index, output_file_dir, in_statement_modulu=0.7, max_layer_num=12):
        _, statement_attention = self.get_statement_attention(tokens,
                                                              statement_index,
                                                              in_statement_modulu, max_layer_num)
        with open(output_file_dir, 'a') as f:
            for statement in statement_attention:
                token = statement['token']
                attention = statement['attention']
                f.write(str(token) + '\n')
                f.write(str(attention) + '\n')


def token_reader(statement_index):
    output_dir = OUTPUT_DIR + '/attentions'
    tokens = []
    with open(output_dir + "/" + str(statement_index) + '/tokens', 'r', encoding='utf-8') as f:
        item = "blank_item"
        while True:
            item = f.readline().rstrip('\n')
            if not item:
                break
            tokens.append(item)
    return tokens


def global_token_attention_reader(output_dir="./weights", filename="latest_output"):
    print("start loading attentions...")
    function_num = len(os.listdir(OUTPUT_DIR + '/attentions'))
    sa = StatementAnalyse()
    for filename in range(0, function_num - 1):
        token = token_reader(int(filename))
        sa.token_frequence_attention_map(token, int(filename))
    print("finish loading attentions...")
    outputMap = {}
    for key in sa.tokenMap.keys():
        if 'Ġ' not in key:
            continue
        attention = sa.tokenMap[key]['attention']
        frequence = sa.tokenMap[key]['frequence']
        if key in outputMap.keys():
            current_attention = outputMap[key]['attention']
            current_frequence = outputMap[key]['frequence']
            outputMap[key]['attention'] = \
                (current_frequence * current_attention + attention) / (current_frequence + frequence)
            outputMap[key]['frequence'] = current_frequence + frequence
        else:
            outputMap[key] = {'attention': attention, 'frequence': frequence}
    return outputMap


def my_tf_color_func(dictionary):
    def my_tf_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(0, 0%%, %d%%)" % (random.randint(20, 60))
    return my_tf_color_func_inner


def overall_analyse():
    g = os.walk(OUTPUT_DIR)
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            print(dir_name)


def get_cloud_item(content, key):
    if key not in content.keys():
        return 0
    return int(math.log(content[key]['attention'] * 100000, 2))


def get_cloud(content):
    data = ""
    with open('./word.txt', 'w') as f:
        for key in content.keys():
            if content[key]['frequence'] > 100:
                data += (key[1:] + " \n") * get_cloud_item(content, key)
        f.flush()
        f.write(data)
    f = open(u'./word.txt', 'r').read()
    wordcloud = WordCloud(background_color="white", width=1200, height=960, margin=1,
                          collocations=False, color_func=my_tf_color_func(content)).generate(f)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file('test.png')
    f.close()


def get_item(content, key):
    return content["Ġ" + key]


def output_token_frequence_and_attention():
    print("start loading attentions...")
    function_num = len(os.listdir(OUTPUT_DIR + '/attentions'))
    print(function_num)
    sa = StatementAnalyse()
    x = []
    y = []
    annotation = []
    for filename in range(0, function_num - 1):
        token = token_reader(int(filename))
        sa.token_frequence_attention_map(token, int(filename))
    print("finish loading attentions...")
    outputMap = {}
    for key in sa.tokenMap.keys():
        if key in cpp_keywords:
            attention = sa.tokenMap[key]['attention']
            frequence = sa.tokenMap[key]['frequence']
            if frequence in outputMap.keys():
                current_attention = outputMap[frequence]['attention']
                current_repeat = outputMap[frequence]['repeat']
                outputMap[frequence]['attention'] = \
                    (current_repeat * current_attention + attention) / (current_repeat + 1)
                outputMap[frequence]['repeat'] = current_repeat + 1
            else:
                outputMap[frequence] = {'attention': attention, 'repeat': 1, 'name': key}
    for key in outputMap.keys():
        attention = outputMap[key]['attention']
        if key >= 1:
            x.append(key * 150)
            y.append(attention)
            annotation.append(outputMap[key]['name'])



def generate_token_attention_file():
    output_file_dir=OUTPUT_DIR + '/token_attentions/'
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    print("start loading attentions...")
    attn_dir = OUTPUT_DIR + '/attentions'
    function_num = len(os.listdir(attn_dir))
    sa = StatementAnalyse()
    for filename in range(0, function_num - 1):
        token = token_reader(int(filename))
        print(token)
        sa.token_frequence_attention_map(token, int(filename))
    attentions = sorted(sa.tokenMap.items(), key=lambda item: np.mean(item[1]['attention']), reverse=False)
    with open(output_file_dir +'token_attentions', 'w', encoding='utf-8') as f:
        for token in attentions:
            f.write(str(token[0]) + '  ||  ' + str(token[1]['attention']) + '\n')


def generate_output_statement_attention():
    function_num = len(os.listdir(OUTPUT_DIR + '/attentions'))
    sa = StatementAnalyse()
    attention_list = []
    output_file_dir = OUTPUT_DIR + '/statement_attentions/'
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    print("loading function files...")
    for filename in range(0, function_num - 1):
        token = token_reader(int(filename))
        _, statement_attention = sa.get_statement_attention(token, int(filename))
        attention_list = attention_list + statement_attention
    print(f"attention_list: {attention_list}")
    print("sorting attention")
    sorted_attention_list = sorted(attention_list, key=lambda statement: statement['attention'])
    print(f"sorted_attention_list: {sorted_attention_list}")
    # print("output statement")
    with open(output_file_dir + 'statement_attention', 'a') as f:
        for statement in sorted_attention_list:
            token = ' '.join(statement['token']).replace('Ġ', '')
            if token == "}":
                continue
            attention = statement['attention']
            f.write(str(attention) + '  ||  ' + str(token) + '\n')

    # for layer_num in range(11, 12):
    #     attention_list = []
    #     print("start process single attention layer")
    #     print("loading function files...")
    #     for filename in range(0, function_num - 1):
    #         token = token_reader(int(filename))
    #         _, statement_attention = sa.get_statement_attention(
    #             token, int(filename), single_layer=True, layer_num=int(layer_num))
    #         attention_list = attention_list + statement_attention
    #         sa.output_statement(token, int(filename), OUTPUT_DIR +'/statement_attentions/output_statement/' + str(filename))
    #     print("sorting attention")
    #     sorted_attention_list = sorted(attention_list, key=lambda statement: statement['attention'])
    #     print("output statement")
    #     with open(OUTPUT_DIR +'/statement_attentions/output_statement/statement_attention_layer_' + str(layer_num), 'a') as f:
    #         for statement in sorted_attention_list:
    #             token = ' '.join(statement['token']).replace('Ġ', '')
    #             if token == "}":
    #                 continue
    #             attention = statement['attention']
    #             f.write(str(attention) + '  ||  ' + str(token) + '\n')


if __name__ == '__main__':
    # #生成token权重文件
    # generate_token_attention_file()
    # #生成cpp_keywords关键词权重文件以及柱形图
    # output_token_frequence_and_attention()
    # #生成statement权重文件
    generate_output_statement_attention()


    pass
