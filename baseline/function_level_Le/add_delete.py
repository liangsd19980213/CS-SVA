import os

import pandas as pd
import re


# 移除C++单行和多行注释的函数
def remove_comments(code):
    # 移除单行注释
    code = re.sub(r'//.*', '', code)
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code


# 读取Excel文件
df = pd.read_excel("../../dataset/megavul_simple_cpp_processed.xlsx")

# 初始化add_lines和delete_lines列
df['add_lines'] = ''
df['delete_lines'] = ''

for idx, row in df.iterrows():
    func_before = row['func_before'].split('\n')
    func = row['func'].split('\n')
    diff_func = row['diff_func'].split('\n')

    add_lines = []
    delete_lines = []

    # 创建忽略空格和注释的行列表
    func_before_stripped = [''.join(remove_comments(line).split()) for line in func_before]
    func_stripped = [''.join(remove_comments(line).split()) for line in func]

    line_num_before = 0
    line_num_after = 0

    start_pos_before = 0
    start_pos_after = 0

    for line in diff_func:
        if line.startswith('@@'):
            # Extract the chunk range
            hunk_info = line.split(' ')[1:3]
            start_line_before = int(hunk_info[0].split(',')[0].replace('-', ''))
            start_line_after = int(hunk_info[1].split(',')[0].replace('+', ''))
            line_num_before = start_line_before - 1
            line_num_after = start_line_after - 1
        elif line.startswith('-'):
            line_num_before += 1
            code_line = ''.join(remove_comments(line[1:]).split())
            for i in range(start_pos_before, len(func_before_stripped)):
                if func_before_stripped[i] == code_line:
                    delete_lines.append(str(i + 1))
                    start_pos_before = i + 1
                    break
        elif line.startswith('+'):
            line_num_after += 1
            code_line = ''.join(remove_comments(line[1:]).split())
            for i in range(start_pos_after, len(func_stripped)):
                if func_stripped[i] == code_line:
                    add_lines.append(str(i + 1))
                    start_pos_after = i + 1
                    break
        else:
            line_num_before += 1
            line_num_after += 1

    df.at[idx, 'add_lines'] = ' '.join(add_lines)
    df.at[idx, 'delete_lines'] = ' '.join(delete_lines)

file_path = r'../../dataset/data.xlsx'
# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件 {file_path} 不存在，正在创建...")
else:
    print(f"文件 {file_path} 已存在，将覆盖保存...")

# 写入 Excel 文件
df.to_excel(file_path, index=False)
print(f"结果已保存至 {file_path}")
# # 将结果写回Excel文件
# df.to_excel('C:\\Users\wjy\PycharmProjects\pythonProject1\c_cpp\新建文件夹 (6)\\data.xlsx', index=False)
