from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch


pretrained_model_name_or_path = r'L:/20230903_Llama2/Atom-7B'
model = AutoModelForCausalLM.from_pretrained(Path(f'{pretrained_model_name_or_path}'), device_map='auto', torch_dtype=torch.float16, load_in_8bit=True, offload_folder="offload") #加载模型
model = model.eval() #切换到eval模式
tokenizer = AutoTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'), use_fast=False) #加载tokenizer
tokenizer.pad_token = tokenizer.eos_token  #为了防止生成的文本出现[PAD]，这里将[PAD]重置为[EOS]
input_ids = tokenizer(['<s>Human: 请用中文介绍一下中国\n</s><s>Assistant: '], return_tensors="pt", add_special_tokens=False).input_ids.to('cuda') #将输入的文本转换为token
generate_input = {
    "input_ids": input_ids, #输入的token
    "max_new_tokens": 512,  #最大生成的token数量
    "do_sample": True,      #是否采样
    "top_k": 50,            #采样的top_k
    "top_p": 0.95,          #采样的top_p
    "temperature": 0.3,     #采样的temperature
    "repetition_penalty": 1.3,               #重复惩罚yi
    "eos_token_id": tokenizer.eos_token_id,  #结束token
    "bos_token_id": tokenizer.bos_token_id,  #开始token
    "pad_token_id": tokenizer.pad_token_id   #pad token
}
generate_ids = model.generate(**generate_input) #生成token
text = tokenizer.decode(generate_ids[0]) #将token转换为文本
print(text) #输出生成的文本