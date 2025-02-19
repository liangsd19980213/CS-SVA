class WeightOutputer():
    def __init__(self):
        self.index = 2
        self.tokenMap = {}
        self.outputFileDir = ''

    def set_output_file_dir(self, fileDir):
        self.outputFileDir = fileDir

    def init_tokenMap(self, tokens):
        for token in tokens:
            if token not in self.tokenMap.keys():
                self.tokenMap[token] = {'frequence': 0, 'attention': 0}

    def update_tokenMap(self, attentions, tokens):
        self.init_tokenMap(tokens)
        for i in range(0, len(attentions)):
            frequence = self.tokenMap[tokens[i]]['frequence']
            attention = self.tokenMap[tokens[i]]['attention']
            self.tokenMap[tokens[i]]['attention'] = (attention * frequence + attentions[i].item()) / (frequence + 1)
            self.tokenMap[tokens[i]]['frequence'] += 1

    def generate_attention_map(self):
        self.set_output_file_dir('./weights')
        f = open(self.outputFileDir + "/" + str(self.index), "r")
        item = 'blank_item'
        while True:
            item = f.readline()
            if not item:
                break
            key = item.split(":0.")[0]
            value = float("0." + item.split(":0.")[1])
            self.attentionMap[key] = value
        f.close()

    def output_weight(self, output_filename):
        f = open(self.outputFileDir + "/" + output_filename, "w")
        for k, v in self.tokenMap.items():
            f.write(k + ":" + str(v['attention']) + '\n')
        f.close()

class Statement():
    def __init__(self, tokens, weight_file_dir="./weights", weight_file_name="latest_output", lang="cpp"):
        self.statements = []
        self.tokens = tokens
        self.statement_attention_map = []
        self.weight_file_dir = weight_file_dir
        self.weight_file_name = weight_file_name
        self.tokenIndexList = []
        self.lang = lang

    def calculate_statement_weights(self):
        for statement in self.statements:
            total_weights = 0.0
            for token in statement:
                total_weights += self.attentionMap[token]
            total_weights /= len(statement)
            self.statement_attention_map.append({'statement': statement, 'attention': total_weights})
        return self.statement_attention_map

    def generate_attention_map(self):
        f = open(self.weight_file_dir + "/" + self.weight_file_name, "r")
        item = 'blank_item'
        while True:
            item = f.readline()
            if not item:
                break
            key = item.split(":0.")[0]
            value = float("0." + item.split(":0.")[1])
            self.attentionMap[key] = value

    def merge_cpp_statements(self):
        token_to_code_index = []
        start = self.tokens.index('<s>') + 1

        # 定义语法规则
        block_keywords = ['if', 'else', 'for', 'while', 'switch', 'try', 'catch', 'do']
        end_keywords = ['}', ';', '#endif']
        # special_start_char = 'Ġ'  # 特殊的空白字符表示新的语句开始

        # 初始化 token_to_code_index
        index = start
        current_token_index = []
        while index < len(self.tokens):
            current_token = self.tokens[index]
            if current_token == "</s>":  # 遇到结束标志，退出循环
                if len(current_token_index) == 1:
                    current_token_index.append(index - 1)
                    token_to_code_index.append(current_token_index)
                break

            # # 新的语句开始：以特殊字符 `Ġ` 开头
            # if current_token.startswith(special_start_char):
            #     self.tokens[index] = current_token[1:]  # 去除 'Ġ' 并更新 token 列表
            #     if current_token_index:
            #         current_token_index.append(index - 1)
            #         token_to_code_index.append(current_token_index)
            #     current_token_index = [index]

            # 按语句分割标志初始化索引
            elif current_token in [';', '{', '}', '(', ')'] or current_token in block_keywords:
                if len(current_token_index) == 1:
                    current_token_index.append(index - 1)
                    token_to_code_index.append(current_token_index)
                current_token_index = [index]
            index += 1

        # # 检查是否初始化失败
        # if not token_to_code_index:
        #     print("Token to Code Index 初始化为空，请检查 token 初始化逻辑。")
        #     return [], []

        statements = []
        index = 0
        in_brace = 0
        start = 0

        # 遍历 tokens 分割语句
        while index < len(token_to_code_index):
            current_token = self.tokens[token_to_code_index[index][0]]

            if current_token in ['{', '(']:
                in_brace += 1
            if current_token in ['}', ')']:
                in_brace -= 1

            if current_token in block_keywords:
                if start != index:
                    statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[index][0]])
                    self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[index - 1][1]])
                start = index
                index += 1
                continue

            if current_token in end_keywords:
                statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[index][0]])
                self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[index - 1][1]])
                start = index + 1
                index += 1
                continue

            index += 1

        # 处理剩余语句
        if start < len(token_to_code_index):
            statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[-1][1]])
            self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[-1][1]])

        return statements, self.tokenIndexList


# if __name__ == '__main__':
#     # 示例输入的 tokens，假设这是由某个 C++ 源代码预处理后生成的
#     tokens = [
#         "</s>", "int", "main", "(", ")", "{", "int", "a", "=", "10", ";",
#         "if", "(", "a", ">", "5", ")", "{", "a", "=", "a", "*", "2", ";", "}",
#         "else", "{", "a", "=", "a", "/", "2", ";", "}", "return", "0", ";", "}"
#     ]
#
#     # 确保 tokens 被正确初始化
#     print("=== 原始 Tokens ===")
#     print(tokens)
#
#     # 创建 Statement 类实例
#     statement_processor = Statement(tokens=tokens, lang="cpp")
#
#     # 调用 merge_cpp_statements 函数
#     statements, token_index_list = statement_processor.merge_cpp_statements()
#
#     # 打印调试信息
#     print("\n=== 分割后的语句 ===")
#     if statements:
#         for statement in statements:
#             print(" ".join(statement))
#     else:
#         print("没有分割出任何语句！请检查 merge_cpp_statements 函数的逻辑。")
#
#     print("\n=== 语句索引范围 ===")
#     if token_index_list:
#         for idx_range in token_index_list:
#             print(idx_range)
#     else:
#         print("没有生成语句索引范围！请检查 merge_cpp_statements 函数的逻辑。")

import logging

# class CodeMerger:
#     def __init__(self, tokens):
#         self.tokens = tokens
#         self.tokenIndexList = []  # Assuming this is needed for tracking token ranges
#
#     def merge_cpp_statements(self):
#         # Validate if end token exists
#         if '</s>' not in self.tokens:
#             logging.error("End token '</s>' not found in the token list.")
#             return [], []
#
#         token_to_code_index = []
#         start = self.tokens.index('</s>') + 1
#
#         # Normalize and clean tokens
#         self.tokens = [token.strip().replace('Ġ', '') for token in self.tokens]  # Remove 'Ġ' and strip spaces
#
#         # Define syntax rules
#         block_keywords = {'if', 'else', 'for', 'while', 'switch', 'try', 'catch', 'do'}
#         end_keywords = {'}', ';', '#endif'}
#         index = start
#         current_token_index = []
#
#         while index < len(self.tokens):
#             current_token = self.tokens[index]
#             if current_token == "</s>":
#                 if current_token_index:
#                     current_token_index.append(index - 1)
#                     token_to_code_index.append(current_token_index)
#                 break
#
#             # Initialize index at delimiters
#             if current_token in {';', '{', '}', '(', ')', '[', ']', '#ifdef', '#ifndef'} or current_token in block_keywords:
#                 if current_token_index:
#                     current_token_index.append(index - 1)
#                     token_to_code_index.append(current_token_index)
#                 current_token_index = [index]
#             index += 1
#
#         if not token_to_code_index:
#             logging.warning("Token to Code Index initialization is empty, check token initialization logic.")
#             return [], []
#
#         # Extract statements based on tokens
#         statements = []
#         index = 0
#         start = 0
#         while index < len(token_to_code_index):
#             current_token_index_range = token_to_code_index[index]
#             statements.append(self.tokens[current_token_index_range[0]:current_token_index_range[1]+1])
#             index += 1
#
#         return statements, token_to_code_index
#
# # Usage example
# tokens =  ['<s>', 'int', 'Ġmain', '(', 'int', 'Ġarg', 'n', ',', 'Ġchar', 'Ġ**', 'arg',
#            'v', ',', 'Ġchar', 'Ġ**', 'env', 'p', ')', 'Ġ{', 'Ġset', 're', 'uid', '(',
#            'get', 'eu', 'id', '(),', 'get', 'eu', 'id', '());', 'Ġ', 'Ġ', 'Ġ', 'Ġ',
#            'Ġset', 'reg', 'id', '(', 'get', 'eg', 'id', '(),', 'get', 'eg', 'id', '());',
#            'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġarg', 'v', '[', '0', ']', 'Ġ=', 'ĠWW', 'SY', 'M', 'PA',
#            ';', 'Ġreturn', 'Ġexec', 've', '(', 'WW', 'SY', 'M', 'PA', ',', 'arg', 'v',
#            ',', 'env', 'p', ');', 'Ġ}', '</s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>', '<s>',
#            '<s>', '<s>', '<s>', '<s>', '<s>', '<s>']
# code_merger = CodeMerger(tokens)
# statements, token_indexes = code_merger.merge_cpp_statements()




