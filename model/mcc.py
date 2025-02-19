import os
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import argparse
from more_itertools import chunked

def calculate_weighted_mcc(y_true, y_pred):
    """
    计算加权 Matthews Correlation Coefficient (MCC weighted)
    参数:
    - y_true: 实际标签列表
    - y_pred: 预测标签列表
    返回:
    - weighted_mcc: 加权 MCC 值
    """
    unique_classes = np.unique(y_true)
    total_samples = len(y_true)
    weighted_mcc = 0.0

    for cls in unique_classes:
        binary_y_true = [1 if y == cls else 0 for y in y_true]
        binary_y_pred = [1 if y == cls else 0 for y in y_pred]
        mcc = matthews_corrcoef(binary_y_true, binary_y_pred)
        class_weight = sum(binary_y_true) / total_samples
        weighted_mcc += class_weight * mcc
    return weighted_mcc

def calculate_macro_mcc(y_true, y_pred):
    """
    计算宏平均 Matthews Correlation Coefficient (MCC macro)
    参数:
    - y_true: 实际标签列表
    - y_pred: 预测标签列表
    返回:
    - macro_mcc: 宏平均 MCC 值
    """
    unique_classes = np.unique(y_true)
    mcc_values = []

    for cls in unique_classes:
        binary_y_true = [1 if y == cls else 0 for y in y_true]
        binary_y_pred = [1 if y == cls else 0 for y in y_pred]
        mcc = matthews_corrcoef(binary_y_true, binary_y_pred)
        mcc_values.append(mcc)

    # 计算宏平均 MCC
    macro_mcc = np.mean(mcc_values)
    return macro_mcc

def parse_data(file_path):
    """
    解析文件内容，提取真实标签和预测标签
    文件格式示例：
    1<CODESPLIT>code_fragment<CODESPLIT>logit1<CODESPLIT>logit2<CODESPLIT>logit3<CODESPLIT>logit4
    """
    y_true = []
    y_pred = []

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('<CODESPLIT>')
            if len(parts) >= 6:  # 确保有标签和至少4个 logits
                true_label = int(parts[0])  # 第一个部分是真实标签
                logits = [float(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1])]  # 提取最后四个 logits
                predicted_label = np.argmax(logits)  # 最大值对应的类别作为预测标签
                y_true.append(true_label)
                y_pred.append(predicted_label)
    return y_true, y_pred

def main():

    file_dir = os.path.join('codet5/result/result_token_0.60.txt')
    # file_dir = os.path.join('codebert/result/result_slim_0.60.txt')

    y_true = []
    y_pred = []
    file_path = os.path.join(file_dir)
    print(f"Processing file: {file_path}")
    file_y_true, file_y_pred = parse_data(file_path)
    y_true.extend(file_y_true)
    y_pred.extend(file_y_pred)
    # 计算加权 MCC
    weighted_mcc = calculate_weighted_mcc(y_true, y_pred)
    print(f" weighted MCC: {weighted_mcc:.4f}")
    macro_mcc = calculate_macro_mcc(y_true, y_pred)
    print(f" macro MCC: {macro_mcc:.4f}")

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--test_batch_size', type=int, default=1000)
    # args = parser.parse_args()
    # languages = ['python', 'java']
    # MCC_dict = {}
    #
    # for language in languages:
    #     file_dir = os.path.join(args.results_dir)
    #     y_true = []
    #     y_pred = []
    #
    #     # 遍历结果文件
    #     for file in sorted(os.listdir(file_dir)):
    #         file_path = os.path.join(file_dir, file)
    #         print(f"Processing file: {file_path}")
    #         file_y_true, file_y_pred = parse_data(file_path)
    #         y_true.extend(file_y_true)
    #         y_pred.extend(file_y_pred)
    #
    #     # 计算加权 MCC
    #     weighted_mcc = calculate_weighted_mcc(y_true, y_pred)
    #     print(f"{language} weighted MCC: {weighted_mcc:.4f}")
    #     MCC_dict[language] = weighted_mcc
    #
    # # 输出最终结果
    # for key, val in MCC_dict.items():
    #     print(f"{key} weighted MCC: {val:.4f}")

if __name__ == "__main__":
    main()
