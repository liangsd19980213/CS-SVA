import pandas as pd
from transformers import RobertaTokenizer
from tqdm import tqdm


def remove_annotation(statements):
    """
    清理代码中的注释，包括单行和多行注释。
    """
    in_annotation = False
    for index, statement in enumerate(statements):
        if in_annotation:
            if '*/' in statement:
                statements[index] = statement[statement.find('*/') + 2:].strip()
                in_annotation = False
            else:
                statements[index] = ''
        elif '//' in statement:
            statements[index] = statement[:statement.find('//')].strip()
        elif '/*' in statement:
            in_annotation = True
            statements[index] = statement[:statement.find('/*')].strip()
    return [s for s in statements if s]


def generate_freq_token_from_csv(csv_file, output_file='./token_frequence'):
    """
    从CSV文件的func_before列读取数据，统计代码中的token频率并保存到文件。
    """
    # 加载数据
    try:
        data = pd.read_csv(csv_file)
        if 'func_before' not in data.columns:
            raise KeyError("The column 'func_before' is missing in the CSV file.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    tokenizer = RobertaTokenizer.from_pretrained('codet5')
    token_map = {}

    # 遍历func_before列数据
    for code in tqdm(data['func_before'].dropna(), desc="Processing functions"):
        try:
            statements = [x.strip() for x in code.replace('\t', ' ').split('\n')]
            statements = remove_annotation(statements)
            cleaned_code = ' '.join(statements)
            tokens = tokenizer.tokenize(cleaned_code)
            for token in tokens:
                token = token.lstrip('Ġ')  # 去掉空格前缀
                token_map[token] = token_map.get(token, 0) + 1
        except Exception as e:
            print(f"Error processing code snippet: {e}")

    # 写入结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, freq in sorted(token_map.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{token}  ||  {freq}\n")


def read_token_freq(filename='./token_frequence'):
    """
    读取 token 频率表，并返回字典 {token: 频率}
    """
    token_map = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('  ||  ')
                if len(parts) == 2:  # 确保格式正确
                    token, freq = parts
                    token_map[token] = int(freq)
    except Exception as e:
        print(f"Error reading token frequency file: {e}")
    return token_map


def prune_token(tokens, rate):
    tokenizer = RobertaTokenizer.from_pretrained('codet5')
    if isinstance(tokens, str):  # 🚀 先 tokenize 代码
        code = tokenizer.tokenize(tokens)
    elif not isinstance(tokens, list):
        raise TypeError(f"Expected list or str for code, got {type(tokens)}")

    token_map = read_token_freq()
    tokens = [str(token) for token in code]  # 确保所有 token 是字符串

    freq = [token_map.get(token, 0) for token in code]
    if not freq:
        return []

    threshold = sorted(freq)[int(len(freq) * rate)]
    return [token for token, f in zip(code, freq) if f >= threshold]



if __name__ == '__main__':
    pass
    # csv_file_path = '../dataset/processed_data/train_data.csv'
    # generate_freq_token_from_csv(csv_file_path)
