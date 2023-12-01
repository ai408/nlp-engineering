"""
Usage: https://github.com/Rayrtfr/Llama2-Chinese/blob/main/tools/merge_llama_with_lora.py
CUDA_VISIBLE_DEVICES="3" python merge_llama_with_lora.py \
    --base_model /path/chinese-llama-plus-lora-7b \
    --lora_model ./path/checkpoint-800 \
    --output_type huggingface \
    --output_dir ./path/checkpoint-800-merge
"""
import argparse
import json
import os
import gc
import torch
import sys
sys.path.append("./")

import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=r"L:/20230713_HuggingFaceModel/20230903_Llama2/Llama-2-7b-hf", required=True, type=str, help="Please specify a base_model") # 指定一个base_model
parser.add_argument('--lora_model', default=r"./checkpoint-500/", required=True, type=str,
                    help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.") # 指定LoRA模型
parser.add_argument('--offload_dir', default=None, type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).") # 指定一个临时文件夹
parser.add_argument('--output_type', default='huggingface', choices=['pth', 'huggingface'], type=str,
                    help="save the merged model in pth or huggingface format.") # 保存合并模型的格式，pth或huggingface
parser.add_argument('--output_dir', default='./merge_llama_with_lora_pt_output', type=str) # 保存合并模型的路径

emb_to_model_size = { # 模型大小
    4096: '7B',
    5120: '13B',
    6656: '30B',
    8192: '65B',
}
num_shards_of_models = {'7B': 1, '13B': 2} # 分片数
params_of_models = { # 模型参数
    '7B': {
            "dim": 4096, # 模型维度
            "multiple_of": 256, # 模型维度的倍数
            "n_heads": 32, # 头数
            "n_layers": 32, # 层数
            "norm_eps": 1e-06, # 归一化
            "vocab_size": -1, # 词表大小
        },
    '13B': {
            "dim": 5120, # 模型维度
            "multiple_of": 256, # 模型维度的倍数
            "n_heads": 40, # 头数
            "n_layers": 40, # 层数
            "norm_eps": 1e-06, # 归一化
            "vocab_size": -1, # 词表大小
        },
}


def transpose(weight, fan_in_fan_out): # 转置
    return weight.T if fan_in_fan_out else weight


# 浏览并修改自https://github.com/tloen/alpaca-lora
def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "") # 基础模型
    if k == "model.embed_tokens.weight": # 嵌入权重
        return "tok_embeddings.weight"
    elif k == "model.norm.weight": # 归一化权重
        return "norm.weight"
    elif k == "lm_head.weight": # 语言模型头权重
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"): # 自注意力权重
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"): # 自注意力权重
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"): # 自注意力权重
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"): # 自注意力权重
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"): # 多层感知机权重
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"): # 多层感知机权重
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"): # 多层感知机权重
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"): # 输入归一化权重
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"): # 后注意力归一化权重
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k: # LoRA
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim) # 转置
    )


def save_shards(model_sd, num_shards: int):
    # 增加no_grad上下文管理器
    with torch.no_grad():
        if num_shards == 1: # 分片数为1
            new_state_dict = {} # 新状态字典
            for k, v in model_sd.items(): # 遍历状态字典
                new_k = translate_state_dict_key(k) # 翻译状态字典键
                if new_k is not None: # 不为空
                    if "wq" in new_k or "wk" in new_k: # wq或wk
                        new_state_dict[new_k] = unpermute(v) # 逆置
                    else: # 其它
                        new_state_dict[new_k] = v # 赋值

            os.makedirs(output_dir, exist_ok=True) # 创建文件夹
            print(f"Saving shard 1 of {num_shards} into {output_dir}/consolidated.00.pth") # 保存分片
            torch.save(new_state_dict, output_dir + "/consolidated.00.pth") # 保存分片
            with open(output_dir + "/params.json", "w") as f: # 保存参数
                json.dump(params, f) # 保存参数
        else: # 分片数不为1
            new_state_dicts = [dict() for _ in range(num_shards)] # 新状态字典
            for k in list(model_sd.keys()): # 遍历状态字典键
                v = model_sd[k] # 状态字典值
                new_k = translate_state_dict_key(k) # 翻译状态字典键
                if new_k is not None: # 不为空
                    if new_k == 'tok_embeddings.weight': # 嵌入权重
                        print(f"Processing {new_k}")
                        assert v.size(1) % num_shards == 0
                        splits = v.split(v.size(1) // num_shards, dim=1)
                    elif new_k == 'output.weight': # 语言模型头权重
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)

                    elif new_k == 'norm.weight': # 归一化权重
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'ffn_norm.weight' in new_k: # 多层感知机权重
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'attention_norm.weight' in new_k: # 输入归一化权重
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards

                    elif 'w1.weight' in new_k: # 多层感知机权重
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)
                    elif 'w2.weight' in new_k: # 多层感知机权重
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1) // num_shards, dim=1)
                    elif 'w3.weight' in new_k: # 多层感知机权重
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)

                    elif 'wo.weight' in new_k: # 自注意力权重
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1) // num_shards, dim=1)

                    elif 'wv.weight' in new_k: # 自注意力权重
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0) // num_shards, dim=0)

                    elif "wq.weight" in new_k or "wk.weight" in new_k: # 自注意力权重
                        print(f"Processing {new_k}")
                        v = unpermute(v)
                        splits = v.split(v.size(0) // num_shards, dim=0)
                    else: # 其它
                        print(f"Unexpected key {new_k}")
                        raise ValueError
                    for sd, split in zip(new_state_dicts, splits): # 遍历新状态字典和分片
                        sd[new_k] = split.clone() # 复制
                        del split # 删除
                    del splits # 删除
                del model_sd[k], v # 删除
                gc.collect()  # 强制垃圾回收

            os.makedirs(output_dir, exist_ok=True) # 创建文件夹
            for i, new_state_dict in enumerate(new_state_dicts): # 遍历新状态字典
                print(f"Saving shard {i + 1} of {num_shards} into {output_dir}/consolidated.0{i}.pth") # 保存分片
                torch.save(new_state_dict, output_dir + f"/consolidated.0{i}.pth") # 保存分片
            with open(output_dir + "/params.json", "w") as f: # 保存参数
                print(f"Saving params.json into {output_dir}/params.json") # 保存参数
                json.dump(params, f) # 保存参数


if __name__ == '__main__':
    args = parser.parse_args() # 解析参数
    base_model_path = args.base_model # 基础模型路径
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip()) != 0] # LoRA模型路径
    output_dir = args.output_dir # 输出路径
    output_type = args.output_type # 输出类型
    offload_dir = args.offload_dir # 临时文件夹

    print(f"Base model: {base_model_path}") # 基础模型
    print(f"LoRA model(s) {lora_model_paths}:") # LoRA模型

    if offload_dir is not None: # 临时文件夹
        # 加载时进行内存外移，这对于内存较低的机器很有用。
        # 请注意，如果有足够的RAM，请改用原始方法，因为它更快。
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path, # 基础模型路径
            load_in_8bit=False, # 加载8位
            torch_dtype=torch.float16, # float16
            offload_folder=offload_dir, # 临时文件夹
            offload_state_dict=True, # 状态字典
            low_cpu_mem_usage=True, # 低CPU内存使用
            device_map={"": "cpu"}, # cpu
        )
    else:
        # 原始方法，不进行外移
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path, # 基础模型路径
            load_in_8bit=False, # 加载8位
            torch_dtype=torch.float16, # float16
            device_map={"": "cpu"}, # cpu
        )
    print(base_model)

    # 从检查点推断模型大小
    embedding_size = base_model.get_input_embeddings().weight.size(1) # 嵌入大小
    model_size = emb_to_model_size[embedding_size] # 模型大小
    print(f"Peft version: {peft.__version__}") # Peft版本
    print(f"Loading LoRA for {model_size} model") # 加载LoRA模型

    lora_model = None # LoRA模型
    lora_model_sd = None # LoRA模型状态字典
    for lora_index, lora_model_path in enumerate(lora_model_paths): # 遍历LoRA模型
        print(f"Loading LoRA {lora_model_path}") # 加载LoRA模型
        tokenizer = LlamaTokenizer.from_pretrained(lora_model_path) # 加载LoRA模型的tokenizer
        assert base_model.get_input_embeddings().weight.size(0) == len(tokenizer) # 断言

        # if base_model.get_input_embeddings().weight.size(0) != len(tokenizer): # 断言
        #     base_model.resize_token_embeddings(len(tokenizer)) # 重新调整词表大小
        #     print(f"Extended vocabulary size to {len(tokenizer)}") # 扩展词表大小

        first_weight = base_model.model.layers[0].self_attn.q_proj.weight # 第一个权重
        first_weight_old = first_weight.clone() # 第一个权重的克隆

        if hasattr(peft.LoraModel, 'merge_and_unload'): # 如果有merge_and_unload方法，将lora model和base model合并为一个独立的model
            lora_model = PeftModel.from_pretrained(
                base_model, # 基础模型
                lora_model_path, # LoRA模型路径
                device_map={"": "cpu"}, # cpu
                torch_dtype=torch.float16, # float16
            )
            assert torch.allclose(first_weight_old, first_weight) # 断言
            print(f"Merging with merge_and_unload...") # 合并
            base_model = lora_model.merge_and_unload() # 合并并卸载
        else:
            base_model_sd = base_model.state_dict() # 基础模型状态字典
            try:
                lora_model_sd = torch.load(os.path.join(lora_model_path, 'adapter_model.bin'), map_location='cpu') # 加载LoRA模型状态字典
            except FileNotFoundError:
                print("Cannot find lora model on the disk. Downloading lora model from hub...") # 从hub下载LoRA模型
                filename = hf_hub_download(repo_id=lora_model_path, filename='adapter_model.bin') # 从hub下载LoRA模型
                lora_model_sd = torch.load(filename, map_location='cpu') # 加载LoRA模型状态字典

            lora_config = peft.LoraConfig.from_pretrained(lora_model_path) # 加载LoRA模型配置
            lora_scaling = lora_config.lora_alpha / lora_config.r # LoRA缩放
            fan_in_fan_out = lora_config.fan_in_fan_out # fan_in_fan_out
            lora_keys = [k for k in lora_model_sd if 'lora_A' in k] # LoRA键
            non_lora_keys = [k for k in lora_model_sd if not 'lora_' in k] # 非LoRA键

            for k in non_lora_keys: # 遍历非LoRA键
                print(f"merging {k}") # 合并
                original_k = k.replace('base_model.model.', '') # 原始键
                base_model_sd[original_k].copy_(lora_model_sd[k]) # 复制

            for k in lora_keys: # 遍历LoRA键
                print(f"merging {k}") # 合并
                original_key = k.replace('.lora_A', '').replace('base_model.model.', '') # 原始键
                assert original_key in base_model_sd # 断言
                lora_a_key = k # LoRA A键
                lora_b_key = k.replace('lora_A', 'lora_B') # LoRA B键
                base_model_sd[original_key] += (transpose(lora_model_sd[lora_b_key].float() @ lora_model_sd[lora_a_key].float(), fan_in_fan_out) * lora_scaling) # 加法
                assert base_model_sd[original_key].dtype == torch.float16 # 断言

        assert not torch.allclose(first_weight_old, first_weight) # 断言

    tokenizer.save_pretrained(output_dir) # 保存tokenizer

    if output_type == 'huggingface': # huggingface格式
        print("Saving to Hugging Face format...") # 保存为Hugging Face格式
        LlamaForCausalLM.save_pretrained(base_model, output_dir, save_function=torch.save, max_shard_size="2GB")  # max_shard_size表示最大分片大小
    else: # pth格式
        print("Saving to pth format...") # 保存为pth格式

        base_model_sd = base_model.state_dict() # 基础模型状态字典
        del lora_model, base_model, lora_model_sd # 删除

        params = params_of_models[model_size] # 模型参数
        num_shards = num_shards_of_models[model_size] # 分片数
        n_layers = params["n_layers"] # 层数
        n_heads = params["n_heads"] # 头数
        dim = params["dim"] # 维度
        dims_per_head = dim // n_heads # 每个头的维度
        base = 10000.0 # 基数
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)) # 逆频率
        save_shards(model_sd=base_model_sd, num_shards=num_shards) # 保存分片