import pandas as pd
import time
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent)))
import gcn
import numpy as np
from transformers import  AutoTokenizer, AutoModel
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
	try:
		code = np.asarray(row['code'].splitlines())  # 将代码分割为数组
		res = ""
		if output == 'code':
			# 解析 vul_line，并检查索引范围
			vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split() if str(line).isdigit()]) - 1
			vul_lines = vul_lines[(vul_lines >= 0) & (vul_lines < len(code))]  # 检查索引范围
			res = code[vul_lines] if len(vul_lines) > 0 else []
		elif output == 'context':
			vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split() if str(line).isdigit()]) - 1
			vul_lines = vul_lines[(vul_lines >= 0) & (vul_lines < len(code))]  # 检查索引范围

			method_lines = np.arange(len(code))  # 所有行的索引
			code_lines = np.setdiff1d(method_lines, vul_lines)  # 非漏洞行的索引
			res = code[code_lines] if len(code_lines) > 0 else []
		elif output == 'all':
			res = code
	except Exception as e:
		print(f"Error in extract_clean_code: {e}, vul_line: {row.get('vul_line')}")
		res = []
	return filter_code(res)


def extract_method_vuln_code(row):
    code = np.asarray(row['code'].splitlines())
    try:
        vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split() if str(line).isdigit()]) - 1
        vul_lines = vul_lines[(vul_lines >= 0) & (vul_lines < len(code))]
        vuln_code = code[vul_lines] if len(vul_lines) > 0 else []
    except Exception as e:
        print(f"Error in extract_method_vuln_code: {e}, vul_line: {row.get('vul_line')}")
        vuln_code = []
    return filter_code(vuln_code)


def extract_context_scope(row, scope_size=5):
    try:
        code_lines = len(row['code'].splitlines())
        vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split() if str(line).isdigit()])
        vul_lines = vul_lines[(vul_lines >= 0) & (vul_lines < code_lines)]
        context_lines = []
        for line in vul_lines:
            start_scope = max(0, line - scope_size)
            end_scope = min(code_lines - 1, line + scope_size)
            context_lines.extend(range(start_scope, end_scope + 1))
        return sorted(list(set(context_lines)))
    except Exception as e:
        print(f"Error in extract_context_scope: {e}, vul_line: {row.get('vul_line')}")
        return []



def extract_surrounding_context_code(row):
    code = np.asarray(row['code'].splitlines())
    try:
        vuln_lines = np.asarray([int(line) for line in row['surrounding_context'] if isinstance(line, (int, str)) and str(line).isdigit()]) - 1
        vuln_lines = vuln_lines[(vuln_lines >= 0) & (vuln_lines < len(code))]
        vuln_code = code[vuln_lines] if len(vuln_lines) > 0 else []
    except Exception as e:
        print(f"Error in extract_surrounding_context_code: {e}, surrounding_context: {row.get('surrounding_context')}")
        vuln_code = []
    return filter_code(vuln_code)



def extract_surrounding_context_code_wo_vuln(row, granularity):
	code = np.asarray(row['code'].splitlines())
	vul_lines = np.asarray([int(line) for line in row["vul_line"].split()])
	vul_lines = np.asarray(list(
		set(row['surrounding_context']) - set(vul_lines) ))-1
	if len(vul_lines) == 0:
		return ''
	vul_lines = code[vul_lines]
	return filter_code(vul_lines)

def extract_random_context(row):
    code = np.asarray(row['code'].splitlines())
    try:
        vul_lines = np.asarray([int(line) for line in str(row["vul_line"]).split() if str(line).isdigit()]) - 1
        vul_lines = vul_lines[(vul_lines >= 0) & (vul_lines < len(code))]
        method_lines = np.arange(len(code))
        code_lines = np.setdiff1d(method_lines, vul_lines)
        code = code[code_lines] if len(code_lines) > 0 else []
    except Exception as e:
        print(f"Error in extract_random_context: {e}, vul_line: {row.get('vul_line')}")
        code = []
    return filter_code(code)

def extract_codebert_features(text, model, tokenizer, batch_size=64):
	start_time = time.time()
	dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(dev)
	text = text.tolist()
	max_length = 512
	tokens_ids = tokenizer(text, max_length=max_length, padding=True, truncation=True, add_special_tokens=True)

	# print(tokens_ids)

	attention_masks = tokens_ids['attention_mask']
	tokens_ids = tokens_ids['input_ids']

	batch_size = 32

	# wrap tensors
	train_data = TensorDataset(torch.tensor(tokens_ids), torch.tensor(attention_masks),
							   torch.tensor([1] * len(tokens_ids)))

	# dataLoader for train set
	train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)

	# for tokens in tokens_ids:
	# 	print(tokenizer.decode(tokens))

	features = []

	for step, batch in enumerate(train_dataloader):

		# print('Step:', step + 1)

		sent_ids, masks, labels = batch
		sent_ids = sent_ids.to(dev)
		masks = masks.to(dev)

		if step == 0:
			features = model(sent_ids, attention_mask=masks)[0][:, 0, :].squeeze().detach().cpu().numpy().tolist()
		else:
			features.extend(model(sent_ids, attention_mask=masks)[0][:, 0, :].squeeze().detach().cpu().numpy().tolist())

	# print(len(features), len(features[0]))

	print('Execution time:', time.time() - start_time, 's.')

	return features

filename = "../../dataset/data.xlsx"
df_method = pd.read_excel(filename)
df_method = df_method.rename(columns={"delete_lines":"vul_line"})

# n_folds = 10

# method_map = create_fold(df_method, 'id', folds=n_folds)

# method_map.to_csv(Data_folder / 'method_map.csv', index=False)



# cvss_cols = ['AV', 'AC', 'PR', 'UI', 'S', 'C','I','A', 'cvss3_severity']
cvss_cols = ['severity']

print('Loaded data')
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
print(len(df_method))
Data_folder = Path("result")  # 将字符串转换为 Path 对象

#############################################################################################
# Whole method

selected_cols = ['cve_id', 'func_before']
selected_cols.extend(cvss_cols)

df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['filtered_code'] = df_tmp[['code']].apply(
	lambda r: extract_clean_code(r, 'all'), axis=1)

df_tmp = df_tmp.drop(columns=["code"])
df_tmp = df_tmp.rename(columns={'cve_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
print(df_tmp)

df_tmp.to_csv(Data_folder / 'method_whole_codebert.csv', index=False)

###########################
# Vuln lines without context in methods

selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['vuln_code'] = df_tmp[['code', 'vul_line']].apply(lambda r: extract_method_vuln_code(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'cve_id': 'key'}).reset_index(drop=True)

codebert_features_code = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features_code

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_lines_without_context_codebert.csv', index=False)

###########################
# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods

scope_size = 6
selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['surrounding_context'] = df_tmp[["code",'vul_line']].apply(
	lambda r: extract_context_scope(r, scope_size=scope_size), axis=1)
df_tmp['context_code'] = df_tmp[['code', 'surrounding_context']].apply(
	lambda r: extract_surrounding_context_code(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'vul_line',  'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'cve_id': 'key'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_lines_with_surrounding_context_codebert.csv', index=False)

###########################
# Context only in methods

selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['filtered_code'] = df_tmp[['code', 'vul_line']].apply(
	lambda r: extract_clean_code(r,  'context'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line'])
df_tmp = df_tmp.rename(columns={'cve_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder /'method_context_only_codebert.csv', index=False)


###########################
# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods

scope_size = 6
selected_cols = ['cve_id', 'func_before', 'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['surrounding_context'] = df_tmp[["code",'vul_line']].apply(
	lambda r: extract_context_scope(r, scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context', 'vul_line']].apply(
	lambda r: extract_surrounding_context_code_wo_vuln(r, granularity='method'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'vul_line', 'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'id': 'key'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_surrounding_only_codebert.csv', index=False)

###########################
# Context only with the same size as the vulnerable statements in methods

selected_cols = ['cve_id', 'func_before',  'vul_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp = df_tmp.rename(columns={"func_before":"code"})
df_tmp['vul_line'] = df_tmp['vul_line'].fillna("").astype(str)
df_tmp['filtered_code'] = df_tmp[['code',  'vul_line']].apply(
	lambda r: extract_random_context(r), axis=1)

df_tmp = df_tmp.drop(columns=['code' , 'vul_line'])
df_tmp = df_tmp.rename(columns={'method_change_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)

codebert_features = extract_codebert_features(df_tmp['code'].values, model, tokenizer, batch_size=16)
df_tmp['codebert'] = codebert_features

df_tmp = df_tmp.drop(columns=['code'])

print(len(df_tmp), df_tmp.columns)
df_tmp.to_csv(Data_folder / 'method_random_context_codebert.csv', index=False)