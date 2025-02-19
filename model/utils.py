import token_frequence
import random
import logging
import os
import csv
from prune import Code_Reduction
from weights import WeightOutputer, Statement
from io import open
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# 定义全局变量
low_rated_tokens = None  # 只在内存中存储一次

current_limit = csv.field_size_limit()  # 获取当前的限制
csv.field_size_limit(current_limit)  # 使用当前限制值，不超出范围
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a  # func_before
        # self.text_b = text_b  # diff_func
        self.label = label    # severity



class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 2:
                    continue
                lines.append(line)
            return lines


class VulnerabilityProcessor(DataProcessor):
    """Processor for the vulnerability assessment task."""

    def get_train_examples(self, data_dir, train_file):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return (self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test"),
            self._read_tsv(os.path.join(data_dir, test_file)))

    def get_labels(self):
        return ["0", "1", "2", "3"]  # Severity levels: Low, Medium, High, Critical

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = line[0]  # Severity label
            text_a = line[1]  # func_before
            # text_a = line[4]  # func_before
            # text_b = line[3]  # diff_func
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer, output_mode='classification',
                                     cls_token_at_end=False, pad_on_left=False,
                                     cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                     sequence_a_segment_id=0, sequence_b_segment_id=1,
                                     cls_token_segment_id=1, pad_token_segment_id=0,
                                     mask_padding_with_zero=True, prune_strategy='None', lang='cpp'):
    """Converts a set of examples to features for model input."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    global origin_code_length
    origin_code_length = 0
    global pruned_code_length
    pruned_code_length = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        origin_code_length += len(example.text_a)
        example.text_a = delete_code_pattern(example.text_a, prune_strategy, lang)
        pruned_code_length += len(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        # tokens_b = tokenizer.tokenize(example.text_b) if example.text_b else None

        # if tokens_b:
        #     origin_code_length += len(tokens_a)
        #     tokens_a = delete_code_pattern(tokens_a, prune_strategy, lang)
        #     pruned_code_length += len(tokens_a)
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # else:
        #     if len(tokens_a) > max_seq_length - 2:
        #         tokens_a = tokens_a[:(max_seq_length - 2)]
        # if len(tokens_a) > max_seq_length - 2:
        #     tokens_a = tokens_a[:(max_seq_length - 2)]

        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        tokens = [cls_token] + tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * (len(tokens_a) + 2)

        # if tokens_b:
        #     tokens += tokens_b + [sep_token]
        #     segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))
        with open('./pruned_rate', 'w') as f:
            f.write(str(origin_code_length) + '\n')
            f.write(str(pruned_code_length) + '\n')
            f.write(str(pruned_code_length / origin_code_length))
    return features

def generate_low_rated_tokens(token_file_dir='codet5/output/token_attentions/low_rated_word', pruned_rate=0.5):
    low_rated_tokens = []
    with open(token_file_dir, 'r') as f:
        token = 'blank_item'
        while True:
            token = f.readline()
            if not token:
                break
            low_rated_tokens.append(token.replace('\n', ''))
    return low_rated_tokens

def get_low_rated_tokens(filepath='codet5/output/token_attentions/low_rated_word'):
    global low_rated_tokens  # 声明使用全局变量
    if low_rated_tokens is None:  # 仅在第一次调用时读取文件
        if os.path.exists(filepath):
            print("Loading existing low-rated tokens...")
            with open(filepath, 'r', encoding='utf-8') as f:
                low_rated_tokens = set(f.read().splitlines())  # 存储为集合，去重
        else:
            print("Generating low-rated tokens...")
            low_rated_tokens = generate_low_rated_tokens(filepath)  # 调用生成函数
    return low_rated_tokens

def delete_code_pattern(code, strategy='None', lang='cpp', **kwargs):
    result = ''
    if strategy == 'trim':
        rate = 0.6
        return code[:int(len(code)*rate)]
    elif strategy == 'frequence':
        rate = 0.60
        result = ' '.join(token_frequence.prune_token(code,rate))
    elif strategy == 'slim':
        reduction = Code_Reduction(code, lang='cpp')
        result = reduction.prune()
    elif strategy == 'token':
        rate = 0.60
        low_rated_tokens = get_low_rated_tokens()
        result = ' '.join(prune_tokens(code, rate, low_rated_tokens))
    elif strategy == 'random':
        rate = 0.4
        if 'rate' in kwargs.keys():
            rate = kwargs['rate']
        result = random_prune_code_with_ratio(code, rate)
    elif strategy == 'None':
        return code
    else:
        return code
    return result


def assimilate_code_string_and_integer(code, string_mask=" string ", number_mask="10"):
    # 处理双引号字符串，包括多行字符串
    def replace_double_quotes(code):
        # 匹配普通双引号字符串（支持转义符号的处理）
        code = re.sub(r'"(\\.|[^"\\])*"', string_mask, code)
        # 匹配 C++ 原始字符串 R"(...)"
        code = re.sub(r'R"\(.*?\)"', string_mask, code, flags=re.DOTALL)
        return code
    # 处理单引号字符
    def replace_single_quotes(code):
        # 匹配单引号包裹的字符，例如 'a', '\n'
        code = re.sub(r"'(\\.|[^'\\])'", string_mask, code)
        return code
    # 替换字符串（普通双引号和原始字符串）
    code = replace_double_quotes(code)
    # 替换字符字面值
    code = replace_single_quotes(code)
    # 将数字替换为占位符
    tokens = code.split(" ")
    for i in range(0, len(tokens)):
        if is_number(tokens[i]):
            tokens[i] = number_mask
    code = " ".join(tokens)
    return code




def random_prune_code_with_ratio(code, rate):
    def random_select_tokens(tokens, ratio):
        pruned_index = sorted(random.sample(range(0, len(tokens)), int(len(tokens) * rate)), reverse=True)
        for index in pruned_index:
            del tokens[index:index+1]
        return tokens
    tokens = code.split(' ')
    result = random_select_tokens(tokens, rate)
    return ' '.join(result)

import re
def prune_tokens(code, rate, low_rated_tokens=[]):
    def camel_case_split(token):
        """将 CamelCase 和 snake_case 拆分成子词"""
        RE_WORDS = re.compile(r'''
        [A-Z]+(?=[A-Z][a-z]) |  # 匹配缩写，例如"XML"中的"XML"
        [A-Z]?[a-z]+ |          # 匹配单个单词，例如"Example"
        [A-Z]+ |                # 匹配大写单词，例如"T" (通常是缩写)
        \d+ |                   # 匹配数字
        [^\u4e00-\u9fa5a-zA-Z0-9]+  # 匹配非字母数字字符（符号）
        ''', re.VERBOSE)
        return RE_WORDS.findall(token)

    def is_not_low_rated_token(token):
        """检查 token 是否在低频 token 列表中"""
        return token if token not in low_rated_tokens else ''

    tokens = code.split(' ')  # 按空格分割代码
    result = []
    for token in tokens:
        snake_case_parts = token.split('_')  # 先处理 snake_case
        camel_case_parts = [subword for part in snake_case_parts for subword in camel_case_split(part)]  # 再处理 CamelCase
        filtered_parts = list(map(is_not_low_rated_token, camel_case_parts))  # 过滤低频 token
        result.append(''.join(filtered_parts))  # 重新拼接
    # 依据简化率进行裁剪
    result = result[:int(len(result) * (1 - rate))]  # 保留 1-rate 的 token
    return result


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





def output_weights(attentions, tokens, modelname, output_dir="./weights", output_statement=True, output_words=True, lang="cpp", outputFileIndex=0):
    wo = WeightOutputer()
    wo.set_output_file_dir(output_dir)
    # global outputFileIndex
    for j in range(0, len(attentions[0])):
        statementGenerator = Statement(tokens[j], lang='cpp')
        statements, tokenIndexList = statementGenerator.merge_cpp_statements()
        # print(statements)
        for i in range(len(attentions)-1, len(attentions)):
            if output_statement:
                output_layer_attention(i, attentions[i][j], tokenIndexList, modelname + "/attentions/" + str(outputFileIndex))
        if output_statement:
            output_tokens(modelname + "/attentions/" + str(outputFileIndex), tokens[j])

        if output_words:
            if outputFileIndex % 100 == 50:
                logger.info("start output token weights into file" + str(outputFileIndex))
                wo.output_weight(str(outputFileIndex))
                logger.info("finish output token weights")
        outputFileIndex += 1

def output_layer_attention(layer, attentions, tokenIndexList, output_file_dir):
    layer_attention_list = []
    for statement_range in tokenIndexList:
        statement_start = statement_range[0]
        statement_end = statement_range[1]
        current_attention_list = [0.0] * len(attentions[0][0])
        for head in attentions:
            for i in range(0, len(head)):
                for j in range(statement_start, statement_end + 1):
                    current_attention_list[j] += head[i][j]
        for i in range(0, len(current_attention_list)):
            current_attention_list[i] = current_attention_list[i] / len(attentions[0][0]) / len(attentions)
        layer_attention_list.append(current_attention_list)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir + '/layer_' + str(layer), 'w') as f:
        for statement in layer_attention_list:
            f.write(str(statement) + '\n')
    return layer_attention_list

def output_tokens(output_file_dir, tokens):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir + "/tokens", 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write("%s\n" % token)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "vulnerability":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "vulnerability": VulnerabilityProcessor,
}

output_modes = {
    "vulnerability": "classification",
}

# if __name__ == '__main__':
#     processors = VulnerabilityProcessor()
#     examples = processors.get_train_examples("../dataset/processed", "train_processed.txt")
#     # 确保输出的数据格式正确
#     print(f"First example: {examples[0]}")  # 显示第一个训练样本
#     # 如果是一个复杂的对象，打印它的类型
#     print(f"Type of first example: {type(examples[0])}")


# if __name__ == '__main__':
#     cpp_code = """
# static BOOL ntlm_av_pair_check(NTLM_AV_PAIR* pAvPair, size_t cbAvPair)
# {
# if (!pAvPair || cbAvPair < sizeof(NTLM_AV_PAIR))
# return FALSE;
# return cbAvPair >= ntlm_av_pair_get_next_offset(pAvPair);
# }
#         """
#     # 测试不同策略
#     strategies = ['method-return', 'trim', 'slim', 'variable', 'loop', 'token', 'random', 'None']
#     strategy = strategies[2]
#     print("原始代码:")
#     print(cpp_code)
#     print("\n测试不同策略的输出:\n")
#     print(f"策略: {strategy}")
#     # 调用 delete_code_pattern
#     processed_code = delete_code_pattern(cpp_code, strategy=strategy, lang='cpp')
#     print(processed_code)
    # print("-" * 80)
    # processed_code = split_cpp_statements(cpp_code)
    # print("原始代码:")
    # print(cpp_code)
    # print("处理代码:")
    # print(processed_code)
