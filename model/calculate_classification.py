from matplotlib import pyplot as plt
import operator
import re
from collections import defaultdict

OUTPUT_DIR = 'codet5/output/statement_attentions'

def parse_statement_attention(filepath):
    """
    从文件中解析语句并计算每种语句类型的平均注意力值。
    """
    attention_data = defaultdict(list)  # 存储每种语句类型的注意力值列表

    # 定义正则表达式，匹配函数调用
    function_call_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\s*(::\s*[a-zA-Z_][a-zA-Z0-9_]*)?\s*\(.*\)\s*;?$'

    # 定义正则表达式，匹配指针操作
    pointer_patterns = [
        r'\*\s*[a-zA-Z_][a-zA-Z0-9_]*',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*&\s*[a-zA-Z_][a-zA-Z0-9_]*',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*\+\+',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*--',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*',
        r'\[\s*\]',
        r'new\s+[a-zA-Z_][a-zA-Z0-9_]*',
        r'\bm?alloc\s*\(.*\)',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*==\s*nullptr',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*!=\s*nullptr'
    ]
    combined_pointer_pattern = '|'.join(pointer_patterns)

    # 定义正则表达式，匹配常见的表达式形式
    expression_patterns = [
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-\*/%]=\s*.+;',
        r'.+\s*[\+\-\*/%]\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(.+\);',
        r'.+\s*[\&\|\^]=\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-]{2};',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*\?\s*.+:\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\<\>\!\=]=\s*.+;',
        r'.+\s*[\&\|\^]{1,2}\s*.+;',
        r'[a-zA-Z_][a-zA-Z0-9_]*\s*(\.\s*|\->\s*).+;'
    ]
    combined_expression_pattern = '|'.join(expression_patterns)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # 分割注意力值和语句内容
                statement = line.split('  ||  ')[1]
                weight = float(line.split('  ||  ')[0])

                # 根据语句内容判断类型
                if 'for' in statement:
                    attention_data['for'].append(weight)
                elif 'while' in statement:
                    attention_data['while'].append(weight)
                elif 'do' in statement:
                    attention_data['do'].append(weight)
                elif 'if' in statement:
                    attention_data['if'].append(weight)
                elif 'else' in statement:
                    attention_data['else'].append(weight)
                elif 'switch' in statement:
                    attention_data['switch'].append(weight)
                elif 'case' in statement:
                    attention_data['case'].append(weight)
                elif 'try' in statement:
                    attention_data['try'].append(weight)
                elif 'catch' in statement:
                    attention_data['catch'].append(weight)
                elif 'throw' in statement:
                    attention_data['throw'].append(weight)
                elif 'return' in statement:
                    attention_data['return'].append(weight)
                elif 'struct' in statement:
                    attention_data['struct'].append(weight)
                elif 'class' in statement:
                    attention_data['class'].append(weight)
                elif '=' in statement:
                    attention_data['variable'].append(weight)
                elif re.match(function_call_pattern, statement.strip()):
                    attention_data['function'].append(weight)
                elif re.match(combined_pointer_pattern, statement.strip()):
                    attention_data['pointer'].append(weight)
                elif re.match(combined_expression_pattern, statement.strip()):
                    attention_data['expression'].append(weight)
                else:
                    attention_data['others'].append(weight)

            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue

    # 计算每种语句类型的平均注意力值
    cpp_statement_classification_map = {
        key: sum(values) / len(values) if values else 0
        for key, values in attention_data.items()
    }

    return cpp_statement_classification_map

def visualize_results(result):
    # 排序结果，根据注意力值降序排序
    sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    # 提取 x 和 y 数据
    x_labels = [item[0] for item in sorted_result]  # 分类名称
    y_values = [item[1] for item in sorted_result]  # 注意力值

    # 保存结果到文件
    with open(OUTPUT_DIR + '/statement_classification.txt', 'w') as f:
        for label, value in sorted_result:
            f.write(f"{label}: {value}\n")

    # 可视化结果
    plt.figure(figsize=(12, 6))  # 设置画布大小
    plt.bar(x_labels, y_values, align='center')  # 绘制柱状图
    plt.xticks(rotation=45, ha='right')  # 分类名称旋转，便于阅读
    plt.ylabel('Attention Weight')  # 设置 Y 轴标签
    plt.xlabel('Statement Type')  # 设置 X 轴标签
    plt.title('C++ Statement Attention Weights')  # 设置图标题
    plt.tight_layout()  # 自动调整布局
    plt.show()


def reduce_statement(statement_list, range_list):
    range_list = sorted(range_list)
    start = 0
    end = 0
    result = []
    for r in range_list:
        start = end
        end = r
        sum = 0
        for line in statement_list:
            if line < end and line >= start:
                sum += 1
        result.append({'range': str(start) + '-' + str(end), 'number': sum})
    start = range_list[-1]
    sum = 0
    for line in statement_list:
        if line > start:
            sum += 1
    result.append({'range': str(start) + '-' + str(end), 'number': sum})
    return result

if __name__ == '__main__':
     # 调用示例
     file_path = 'codet5/output/statement_attentions/statement_attention'  # 请替换为实际文件路径
     cpp_statement_classification_map = parse_statement_attention(file_path)

     # 打印结果
     print(f"cpp_statement_classification_map: {cpp_statement_classification_map}")

     visualize_results(cpp_statement_classification_map)