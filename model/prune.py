import copy

import numpy as np
import pandas as pd

import calculate_classification
import cpp_statement as cp
import re

cpp_statement_classification_map = {}
lowest_ranked_token = []
DATA_PATH = '../dataset/processed_data/train_data.csv'
STATEMENT_ATTENTIONS_PATH = 'codet5/output/statement_attentions/statement_attention'
LOW_RATED_TOKEN_PATH = 'codet5/output/token_attentions/low_rated_word'
# STATEMENT_ATTENTIONS_PATH = 'codebert/output/statement_attentions/statement_attention'
# LOW_RATED_TOKEN_PATH = 'codebert/output/token_attentions/low_rated_word'

def split_cpp_statements_by_line(code):
    statements = []
    # 按换行符分割代码
    lines = code.split("\n")
    for line in lines:
        stripped_line = line.strip()  # 去除首尾空格
        if stripped_line:  # 过滤空行
            statements.append(stripped_line)
    return statements

def read_func_before_column(file_path, chunksize=1000):
    """
    逐块读取 CSV，减少内存占用，提高处理效率
    """
    func_before_list = []
    for chunk in pd.read_csv(file_path, usecols=['func_before'], chunksize=chunksize):
        func_before_list.extend(chunk['func_before'].dropna().tolist())
    return func_before_list

def split_cpp_statements(code):
    # 匹配多行的控制结构的开头，包括条件编译语句
    control_pattern = re.compile(
        r'(#if\b|#ifdef\b|#ifndef\b|#endif\b|if\b|else if\b|else\b|for\b|while\b|do\b|switch\b)', re.MULTILINE
    )
    # 存储控制语句和非控制语句的列表
    statements = []
    index = 0
    length = len(code)
    max_iterations = 10_000  # 防止死循环设置最大迭代次数
    iterations = 0
    while index < length:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError("Exceeded maximum iterations, possible infinite loop.")
        match = control_pattern.search(code, index)
        if match:
            start = match.start()
            # 添加控制结构前的内容为非控制语句
            if start > index:
                statements.append(code[index:start].strip())

            # 获取匹配的控制结构类型
            control_type = match.group()
            end = match.end()
            # 设置大括号和小括号计数器
            brace_count = 0
            paren_count = 0
            has_braces = False
            # 匹配完整的小括号对（用于 for, if, while, switch 等结构）
            if control_type in {"if", "else if", "for", "while", "switch"}:
                # 处理小括号
                while end < length:
                    char = code[end]
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            end += 1  # 完整的小括号匹配结束
                            break
                    end += 1
                # 检查大括号是否存在以确定块范围
                while end < length:
                    char = code[end]
                    if char == '{':
                        brace_count += 1
                        has_braces = True
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and has_braces:
                            end += 1  # 包括右大括号
                            break
                    elif char == ';' and not has_braces:
                        # 单行语句到分号为止
                        end += 1
                        break
                    end += 1
                # 检查是否有 `else` 块并继续匹配
                while end < length:
                    # 匹配紧跟其后的 "else" 或 "else if"
                    else_match = re.match(r'\s*else\b', code[end:])
                    if else_match:
                        end += else_match.end()
                        # 检查是否是 "else if" 结构
                        if re.match(r'\s*if\b', code[end:]):
                            end += 2  # 跳过 "if"
                            # 处理 "else if" 的小括号
                            while code[end] != ')' and end < length:
                                if code[end] == '(':
                                    paren_count += 1
                                elif code[end] == ')':
                                    paren_count -= 1
                                    if paren_count == 0:
                                        end += 1
                                        break
                                end += 1
                        # 处理大括号
                        while end < length:
                            char = code[end]
                            if char == '{':
                                brace_count += 1
                                has_braces = True
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0 and has_braces:
                                    end += 1
                                    break
                            elif char == ';' and not has_braces:
                                end += 1
                                break
                            end += 1
                    else:
                        break
            elif control_type == "do":
                # do-while 结构处理
                while end < length:
                    char = code[end]
                    if char == '{':
                        brace_count += 1
                        has_braces = True
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and has_braces:
                            end += 1
                            break
                    elif char == ';' and not has_braces:
                        end += 1
                        break
                    end += 1
                # 找到 "while" 关键字，结束 do-while
                while end < length and not re.match(r'\bwhile\b', code[end:]):
                    end += 1
                # 跳过 "while" 和分号
                while end < length and code[end] != ';':
                    end += 1
                end += 1
            elif control_type in {"#ifdef", "#ifndef", "#if"}:
                # 匹配 #ifdef, #ifndef, #if 到 #endif 结构，包括嵌套情况
                nested_count = 1
                end = match.end()
                while end < length:
                    nested_match = re.match(r'#(ifdef|ifndef|if)\b', code[end:])
                    endif_match = re.match(r'#endif\b', code[end:])
                    if nested_match:
                        nested_count += 1
                        end += nested_match.end()
                    elif endif_match:
                        nested_count -= 1
                        end += endif_match.end()
                        if nested_count == 0:
                            break
                    else:
                        end += 1
            # 将完整的控制结构框架添加到控制语句列表，而不包含内部内容
            segment = code[start:end].strip()
            if segment not in statements:  # 避免重复
                statements.append(segment)
            # 更新索引，继续查找
            index = end
        else:
            # 如果没有匹配到控制结构，将剩余部分视为非控制语句
            remaining_code = code[index:].strip()
            if remaining_code:  # 避免空白行加入
                statements.append(remaining_code)
            break
    return statements


def get_token_attention():
    with open(LOW_RATED_TOKEN_PATH, 'r') as f:
        for token in f.readlines():
            lowest_ranked_token.append(token.replace('\n', ''))
get_token_attention()

def camel_case_split_cpp(s):
    import re
    # 先处理下划线分隔
    s = '_'.join(s.split('_'))
    # 正则处理 CamelCase 和其他情况
    RE_WORDS = re.compile(r'''
        [A-Z]+(?=[A-Z][a-z]) |  # 大写单词（后接小写）
        [A-Z]?[a-z]+         |  # 小写单词
        [A-Z]+               |  # 纯大写单词
        \d+                  |  # 数字
        [^\w]+                 # 非单词字符
    ''', re.VERBOSE)
    return RE_WORDS.findall(s)

def underline(str):
    return str.split('_')

def get_cpp_statement_classification(statement, cpp_statement_classification_map):
    # 判断语句类型
    if cp.is_for_statement(statement):
        return 'for', cpp_statement_classification_map['for']
    elif cp.is_do_statement(statement):
        return 'do', cpp_statement_classification_map['do']
    elif cp.is_if_statement(statement):
        return 'if', cpp_statement_classification_map['if']
    elif cp.is_return_statement(statement):
        return 'return', cpp_statement_classification_map['return']
    elif cp.is_struct_declaration(statement):
        return 'struct', cpp_statement_classification_map['struct']
    elif cp.is_variable(statement):
        return 'variable', cpp_statement_classification_map['variable']
    else:
        return "None", 0.0001


# def get_cpp_statement_classification(statement, cpp_statement_classification_map):
#     # 判断语句类型
#     if cp.is_for_statement(statement):
#         return 'for', cpp_statement_classification_map['for']
#     elif cp.is_while_statement(statement):
#         return 'while', cpp_statement_classification_map['while']
#     elif cp.is_do_statement(statement):
#         return 'do', cpp_statement_classification_map['do']
#     elif cp.is_if_statement(statement):
#         return 'if', cpp_statement_classification_map['if']
#     elif cp.is_else_statement(statement):
#         return 'else', cpp_statement_classification_map['else']
#     elif cp.is_try_statement(statement):
#         return 'try', cpp_statement_classification_map['try']
#     elif cp.is_return_statement(statement):
#         return 'return', cpp_statement_classification_map['return']
#     elif cp.is_struct_declaration(statement):
#         return 'struct', cpp_statement_classification_map['struct']
#     elif cp.is_variable(statement):
#         return 'variable', cpp_statement_classification_map['variable']
#     elif cp.is_expression(statement):
#         return 'expression', cpp_statement_classification_map['expression']
#     else:
#         return "None", 0.0001


class Code_Reduction():  # self.statement_attention: statement categories' attention. Form as:
    #      [{category: 'if statement', content: 'statement content', attention: 0.01, length: 10}]
    # self.token_attention: token attention. Form as {'a': 0.01, 'b': 0.02}
    def __init__(self, code, lang='cpp', ratio = 0.4, **kwargs):  #change code path(line8,9)
        # get_cpp_statement_classification改成对应map中有的条件
        self.code = code
        self.lang = lang
        self.ratio = ratio
        # self.targetLength = targetLength  targetLength=256
        self.result = []
        self.generate_statements()

    def generate_statements(self):
        cpp_statement_classification_map = {}
        self.statements = []
        # statements = None
        # statements = split_cpp_statements(self.code)

        statements = []
        func_before_list = read_func_before_column(DATA_PATH)
        for code_snippet in func_before_list:
            statement = split_cpp_statements_by_line(code_snippet)
            statements.append(statement)  # Combine results

        for statement in statements:
            cpp_statement_classification_map = calculate_classification.parse_statement_attention(STATEMENT_ATTENTIONS_PATH)
            # print(f"cpp_statement_classification_map: {cpp_statement_classification_map}")
            category, attention = get_cpp_statement_classification(statement,cpp_statement_classification_map)
            current_statement = {'category': category, 'content': statement,
                                 'length': len(statement), 'attention': attention}
            self.statements.append(current_statement)
        # print(f"self.statements: {self.statements}")

    def generate_statement_attention(self, attention_file_dir='./statement_attention'):
        self.statement_attention = []
        # self.statements = self.statements(key=lambda item: item['attention'], reverse=True)

    def generate_token_attention(self, attention_file_dir='./token_attention'):
        self.token_attention = {}

    def get_statement_attention(self, statement):
        pass

    def prune_lowest_ranked_token(self, statements, prune_num):
        result = []
        # check pruning items
        candidate = []
        for statement in statements:
            for token in statement:
                if token in lowest_ranked_token:
                    attention_pos = lowest_ranked_token.index(token)
                    if len(candidate) <= prune_num:
                        candidate.append(attention_pos)
                    elif attention_pos < max(candidate) and attention_pos not in candidate:
                        candidate.remove(max(candidate))
                        candidate.append(attention_pos)
        # prune phase
        pruned_num = 0
        need_check = True
        candidate = [lowest_ranked_token[x] for x in candidate]
        for statement in statements:
            if not need_check:
                result.append(statement)
                continue
            current_statement = []
            for token in statement:
                if token in candidate and need_check:
                    pruned_num += 1
                    if pruned_num >= prune_num:
                        need_check = False
                else:
                    current_statement.append(token)
                continue
            result.append(current_statement)
        return result


    def zero_one_backpack(self):
        # after the 0-1 backpack problem solution get the chosen statements we prefer to reduce tokens insteam of add
        # tokens so we choose to increase the target length by the max length of all the statements so that the solution
        # will at least consist of more than one statement than the solution of the previous target length.
        # max_length = 0
        # print(f"self.statements: {self.statements}")
        # for statement in self.statements:
        #     if statement['length'] > max_length:
        #         max_length = statement['length']
        # max_length += self.targetLength
        # dp = [[{'attention': 0.0, 'statements': []}
        #        for i in range(max_length + 1)] for j in range(len(self.statements) + 1)]
        # 计算原始代码的总长度
        total_length = sum(statement['length'] for statement in self.statements)
        target_length = int(total_length * (1 - self.ratio))  # 计算目标长度

        # 动态规划表初始化
        dp = [[{'attention': 0.0, 'statements': []}
               for _ in range(target_length + 1)] for _ in range(len(self.statements) + 1)]

        for i in range(1, len(self.statements) + 1):
            for j in range(1, target_length  + 1):
                current_map = {'attention': dp[i-1][j]['attention'],
                               'statements': copy.deepcopy(dp[i-1][j]['statements'])}
                dp[i][j] = current_map
                if j >= self.statements[i-1]['length']:
                    if dp[i][j]['attention'] < \
                            dp[i-1][j-self.statements[i-1]['length']]['attention'] + self.statements[i-1]['attention']:
                        dp[i][j]['attention'] = dp[i-1][j-self.statements[i-1]
                                                        ['length']]['attention'] + self.statements[i-1]['attention']
                        dp[i][j]['statements'] = copy.deepcopy(
                            dp[i-1][j-self.statements[i-1]['length']]['statements'])
                        dp[i][j]['statements'].append(i-1)
        return dp[-1][-1]['attention'], dp[-1][-1]['statements']


    def prune(self, **kwargs):
        # adding statments: greedy
        total_attention, chosen_statements = self.zero_one_backpack()
        # print(f'total_attention: {total_attention}')
        # print(f'chosen_statements: {chosen_statements}')

        current_length = 0
        result = []
        for statement_index in chosen_statements:
            current_length += self.statements[statement_index]['length']
            result.append(self.statements[statement_index]['content'])
        # print(f'result: {result}')
        # pruned_token_num = current_length - self.targetLength
        # 如果总长度超出目标长度，则进一步剪枝
        total_length = sum(statement['length'] for statement in self.statements)
        target_length = int(total_length * (1 - self.ratio))
        pruned_token_num = current_length - target_length
        if pruned_token_num > 0:
            result = self.prune_lowest_ranked_token(result, pruned_token_num)
        return ' '.join(' '.join(x) for x in result)
