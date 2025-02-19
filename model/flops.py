# import torch
# from transformers import T5EncoderModel, RobertaTokenizer, T5Config
# from fvcore.nn import FlopCountAnalysis
#
# from model import CodeT5ForSequenceClassification
#
# # 通常 CodeT5 使用的是 T5 编码器模型
# # 如果你有预训练的模型路径，确保替换 'path_to_codet5'
# tokenizer = RobertaTokenizer.from_pretrained('codet5')
# encoder = T5EncoderModel.from_pretrained('codet5')
# model = CodeT5ForSequenceClassification(encoder)
# model = torch.load('codet5/slim_0.45/checkpoint-best/pytorch_model.bin', map_location='cuda')
#
# # 将模型设置为评估模式
# model.eval()
#
# # # 加载预训练的 CodeT5 模型
# # config = T5Config.from_pretrained('codet5')
# # model = T5EncoderModel.from_pretrained('codet5', config=config)
# # model.eval()
# #
# # # 创建一个假输入，适应模型的输入需求
# # # 假设输入长度为 512，这应该根据你的模型训练时的设置来调整
# # input_ids = torch.randint(0, 32128, (1, 256))  # 32128 是 T5 模型的词汇大小
# # attention_mask = torch.ones(1, 256)
#
# # 使用 fvcore 的 FlopCountAnalysis 来计算 FLOPs
# inputs = (torch.randint(0, 32128, (1, 141))) #输入长度基线模型：256 简化模型：256*（1-ratio）
# flops = FlopCountAnalysis(model, inputs)
#
# # 输出计算的 FLOPs
# print(f"CodeT5 Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # 输出为 Giga FLOPs
#
import torch
from transformers import T5EncoderModel, RobertaTokenizer
from fvcore.nn import FlopCountAnalysis
from model import CodeT5ForSequenceClassification

# 加载 tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('codet5')
# encoder = T5EncoderModel.from_pretrained('codet5')
# model = CodeT5ForSequenceClassification(encoder)

tokenizer = RobertaTokenizer.from_pretrained('codebert')
encoder = T5EncoderModel.from_pretrained('codebert')
model = CodeT5ForSequenceClassification(encoder)


# 正确加载模型权重
# model = torch.load('codet5/slim_0.45/checkpoint-best/pytorch_model.bin')

model = torch.load('codebert/slim_0.45/checkpoint-best/pytorch_model.bin')

model.to('cuda')  # 确保模型在 GPU 上
model.eval()  # 设置为评估模式

# 生成输入数据，并移动到 CUDA
input_length = 141  # 代码简化后的输入长度
input_ids = torch.randint(0, 32128, (1, input_length)).to('cuda')
attention_mask = torch.ones_like(input_ids).to('cuda')

# 计算 FLOPs
flops = FlopCountAnalysis(model, (input_ids, attention_mask))
# print(f"CodeT5 Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # 输出为 Giga FLOPs

print(f"CodeBERT Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # 输出为 Giga FLOPs
