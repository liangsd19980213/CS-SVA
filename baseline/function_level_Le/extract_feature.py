from pathlib import Path

import pandas as pd
import time
import sys

path = "/".join(sys.path[0].split("/")[:-2])
sys.path.append(path)
import gcn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader


def filter_code(vuln_code):
    code_lines = []

    for code_line in vuln_code:
        if '//' in code_line:
            code_line = code_line[:code_line.find('//')]
        elif '/*' in code_line and '*/' in code_line:
            start_comment_index = code_line.find('/*')
            end_comment_index = code_line.find('*/')

            code_line = code_line[:start_comment_index] + code_line[end_comment_index + 2:]

        code_lines.append(code_line)

    return '\n'.join(code_lines)


def extract_clean_code(row, output='code'):
    # Output options: code, context and all (code + context)
    code = np.asarray(row['code'].splitlines())
    res = ""
    if output == 'code':
        vul_lines = np.asarray([int(line) for line in row["vul_line"].split()]) - 1
        res = code[vul_lines]
    elif output == 'context':
        vul_lines = np.asarray([int(line) for line in row["vul_line"].split()]) - 1
        method_lines = np.asarray(list(range(len(code))))
        method_lines = method_lines.tolist()
        vul_lines = vul_lines.tolist()
        code_lines = np.asarray(
            list(set(method_lines) - set(vul_lines)))
        if len(code_lines) != 0:
            res = code[code_lines]
        else:
            res = ""
    elif output == 'all':
        res = code
    return filter_code(res)


def extract_method_vuln_code(row):
    code = np.asarray(row['code'].splitlines())

    try:
        # 尝试解析 vul_line，如果失败则替换为 [0]
        vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split() if line.isdigit()]) - 1
        # 如果 vul_lines 为空，默认值为 [0]
        if len(vul_lines) == 0:
            vul_lines = np.array([0])
    except:
        # 如果解析失败，设置默认值 [0]
        vul_lines = np.array([0])

    # 确保 vul_lines 不越界
    vul_lines = vul_lines[(vul_lines >= 0) & (vul_lines < len(code))]

    # 提取对应的代码行
    vuln_code = code[vul_lines]

    return filter_code(vuln_code)


def extract_context_scope(row, scope_size=5):
    # 确保 code 存在并分割为行
    code = np.asarray(row['code'].splitlines())
    code_length = len(code)

    # 如果 vul_line 是 NaN 或不存在，则返回空列表
    if pd.isna(row["vul_line"]):
        return []

    try:
        # 处理 vul_line 数据，转换为字符串，过滤非法值，并计算范围
        vul_lines = np.asarray([
            int(line) for line in str(row["vul_line"]).split()
            if line.isdigit() and 0 <= int(line) - 1 < code_length
        ])

        context_lines = []

        for line in vul_lines:
            # 计算上下文范围
            start_scope = max(0, line - scope_size)
            end_scope = min(code_length - 1, line + scope_size)
            context_lines.extend(range(start_scope, end_scope + 1))

        # 返回去重后的上下文行索引
        return sorted(set(context_lines))

    except Exception as e:
        # 如果处理失败，打印错误信息并返回空列表
        print(f"Error processing row: {row} - {e}")
        return []


def extract_surrounding_context_code(row):
    code = np.asarray(row['code'].splitlines())
    # print(row['surrounding_context'])
    # print(set(row['surrounding_context']))
    vuln_lines = np.asarray(list(set(row['surrounding_context']))) - 1
    if len(vuln_lines) == 0:
        return ''

    vuln_code = code[vuln_lines]

    return filter_code(vuln_code)


def extract_surrounding_context_code_wo_vuln(row, granularity):
    code = np.asarray(row['code'].splitlines())
    vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split()])
    vul_lines = np.asarray(list(
        set(row['surrounding_context']) - set(vul_lines))) - 1

    if len(vul_lines) == 0:
        return ''

    vul_lines = code[vul_lines]

    return filter_code(vul_lines)


def extract_random_context(row):
    code = np.asarray(row['code'].splitlines())
    vul_lines = np.asarray([int(line) for line in row["vul_line"].split()]) - 1

    method_lines = np.asarray(list(range(len(code)))) + 1
    method_lines = method_lines.tolist()
    code_lines = np.asarray(
        list(set(method_lines) - set(vul_lines))) - 1

    vuln_lines_len = len(vul_lines)
    if vuln_lines_len > len(code_lines):
        vuln_lines_len = len(code_lines)

    code_lines = np.random.RandomState(vuln_lines_len).choice(code_lines, vuln_lines_len, replace=False)

    if len(code_lines) == 0:
        return ''

    code = code[code_lines]

    return filter_code(code)



filename = "../../dataset/data.xlsx"
df_method = pd.read_excel(filename)

print('Loaded data')
Data_folder = Path("result")  # 将字符串转换为 Path 对象

df_method = df_method.iloc[np.random.RandomState(42).permutation(len(df_method))].reset_index(drop=True)

df_method = df_method.rename(columns={"delete_lines": "vul_line"})

cvss_cols = ['severity']

print(len(df_method))
print(df_method.count())
print(df_method.columns)

selected_cols = ['cve_id', 'func_before']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before": "code"})
df_tmp['filtered_code'] = df_tmp[['code']].apply(
    lambda r: extract_clean_code(r, 'all'), axis=1)

df_tmp = df_tmp.drop(columns=["code"])
df_tmp = df_tmp.rename(columns={'id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
print(df_tmp)

df_tmp.to_csv(Data_folder / 'method_whole.csv', index=False)
# Vuln lines without context in methods
selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before": "code"})
df_tmp['vuln_code'] = df_tmp[['code', 'vul_line']].apply(lambda r: extract_method_vuln_code(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'cve_id': 'key'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_lines_without_context.csv', index=False)

# program slice
# Vuln lines with context in methods
# selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'noisy_lines']
# selected_cols.extend(cvss_cols)
# df_tmp = df_method[selected_cols].copy()
# df_tmp['vuln_code'] = df_tmp[['code', 'context_lines', 'start_line', 'noisy_lines']].apply(lambda r: extract_context_code_method(r), axis=1)
# df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'noisy_lines'])
# df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)


# print(len(df_tmp), df_tmp.columns)
# df_tmp.to_parquet('Data/method_lines_with_all_context.parquet', index=False)

#
# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods
scope_size = 6

selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before": "code"})
df_tmp['surrounding_context'] = df_tmp[["code", 'vul_line']].apply(
    lambda r: extract_context_scope(r, scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context']].apply(
    lambda r: extract_surrounding_context_code(r), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line', 'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'cve_id': 'key'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_lines_with_surrounding_context.csv', index=False)

# Context only in methods
selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before": "code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['filtered_code'] = df_tmp[['code', 'vul_line']].apply(
    lambda r: extract_clean_code(r, 'context'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'cve_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_context_only.csv', index=False)

# program slicing
# Non vuln lines in methods (all scope - program slicing scope)
# selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'noisy_lines']
# selected_cols.extend(cvss_cols)
# df_tmp = df_method[selected_cols].copy()
# df_tmp['vuln_code'] = df_tmp[['code', 'context_lines', 'start_line', 'noisy_lines']].apply(
# 	lambda r: extract_non_vuln_code_method(r), axis=1)
# df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'noisy_lines'])
# df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)


# print(len(df_tmp), df_tmp.columns)
# df_tmp.to_parquet('Data/method_non_vuln.parquet', index=False)

###########################

# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods
scope_size = 6

selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before": "code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['surrounding_context'] = df_tmp[["code", 'vul_line']].apply(
    lambda r: extract_context_scope(r, scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context', 'vul_line']].apply(
    lambda r: extract_surrounding_context_code_wo_vuln(r, granularity='method'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line', 'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'cve_id': 'key'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_surrounding_only.csv', index=False)

###########################
# program slice
# Vuln lines with context in methods
# selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']
# selected_cols.extend(cvss_cols)
# df_tmp = df_method[selected_cols].copy()

# df_tmp['context_code'] = df_tmp[['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']].apply(
# 	lambda r: extract_context_code_method_wo_vuln(r), axis=1)

# df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines'])
# df_tmp = df_tmp.rename(columns={'context_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)


# print(len(df_tmp), df_tmp.columns)
# df_tmp.to_parquet('Data/method_slicing_only.parquet', index=False)

# Context only with the same size as the vulnerable statements in methods
selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before": "code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['filtered_code'] = df_tmp[['code', 'vul_line']].apply(
    lambda r: extract_random_context(r), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'method_change_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_random_context.csv', index=False)
