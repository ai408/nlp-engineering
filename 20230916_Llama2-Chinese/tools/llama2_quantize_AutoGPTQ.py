from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig # 量化配置
from transformers import AutoTokenizer

# 第1部分：量化一个预训练模型
pretrained_model_name = r"L:/20230713_HuggingFaceModel/20230903_Llama2/Llama-2-7b-hf" # 预训练模型路径
quantize_config = BaseQuantizeConfig(bits=4, group_size=128) # 量化配置，bits表示量化后的位数，group_size表示分组大小
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_name, quantize_config) # 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name) # 加载tokenizer

examples = [ # 量化样本
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]
# 翻译：准备examples（一个只有两个键'input_ids'和'attention_mask'的字典列表）来指导量化。这里只使用一个文本来简化代码，但是应该注意，使用的examples越多，量化后的模型就越好（很可能）。
model.quantize(examples) # 执行量化操作，examples提供量化过程所需的示例数据
quantized_model_dir = "./llama2_quantize_AutoGPTQ" # 保存量化后的模型
model.save_quantized(quantized_model_dir) # 保存量化后的模型


# 第2部分：加载量化模型和推理
from transformers import TextGenerationPipeline # 生成文本

device = "cuda:0"
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=device) # 加载量化模型
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device) # 得到pipeline管道
print(pipeline("auto-gptq is")[0]["generated_text"]) # 生成文本