import torch
from transformers import T5ForConditionalGeneration, RobertaForSequenceClassification
from ptflops import get_model_complexity_info

# 选择模型
model_name = "codebert"  # 修改为 "codebert" 测试 CodeBERT

if model_name == "codet5":
    model = T5ForConditionalGeneration.from_pretrained("./trained_model_codet5")
    decoder_start_token_id = model.config.decoder_start_token_id  # 获取解码器的起始标记
elif model_name == "codebert":
    model = RobertaForSequenceClassification.from_pretrained("./trained_model_codebert")
else:
    raise ValueError("Invalid model name. Choose either 'codet5' or 'codebert'.")

# 获取词汇表大小
vocab_size = model.config.vocab_size

# 创建符合范围的 dummy_input 和 decoder_input_ids
sequence_length = 256
batch_size = 1
dummy_input = torch.randint(0, vocab_size - 1, (batch_size, sequence_length), dtype=torch.long)
if model_name == "codet5":
    decoder_input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long) * decoder_start_token_id
else:
    decoder_input_ids = None  # CodeBERT 不需要 decoder_input_ids

# 包装模型的 forward 方法，使其与 ptflops 兼容
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, model_name, decoder_input_ids=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.model_name = model_name
        self.decoder_input_ids = decoder_input_ids

    def forward(self, x):
        # 强制转换输入为 LongTensor
        if x.dtype != torch.long:
            x = x.to(dtype=torch.long)
        # 修正输入范围，确保不超过 vocab_size
        x = x.clamp(0, vocab_size - 1)

        if self.model_name == "codet5":
            # 传递 input_ids 和 decoder_input_ids
            outputs = self.model(input_ids=x, decoder_input_ids=self.decoder_input_ids)
            return outputs.logits
        else:
            # 仅传递 input_ids
            outputs = self.model(input_ids=x)
            return outputs

wrapped_model = ModelWrapper(model, model_name, decoder_input_ids)

# FLOPs 计算
def flops_counter(input_res, model):
    with torch.no_grad():
        flops, params = get_model_complexity_info(
            model,
            input_res=input_res,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=True,
        )
        return flops, params

# 输入维度
input_res = (sequence_length,)  # 输入序列长度

try:
    flops, params = flops_counter(input_res, wrapped_model)
    print(f"FLOPs: {flops}")
    print(f"Number of parameters: {params}")
except RuntimeError as e:
    print(f"FLOPs estimation failed due to: {e}")
