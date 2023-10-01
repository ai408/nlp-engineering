"""
翻译：在文本文件或数据集上对因果语言建模（GPT、GPT-2、CTRL等）的库模型进行微调。
翻译：这是可以通过此脚本微调的hub上的检查点的完整列表：https://huggingface.co/models?filter=text-generation
"""
# 翻译：还可以在自己的因果语言建模任务上调整此脚本。这方面的pointer留在了注释中。
import logging
import math
import os
import sys
import random
from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional, List, Union

import datasets
import evaluate
import torch
from datasets import load_dataset
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import pdb

# 翻译：如果没有安装transformers的最小版本，将出错。自行决定是否删除。check_min_version("4.27.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    翻译：与我们将要微调或从头开始训练的模型/配置/令牌器有关的参数。
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[str] = field(
        default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    load_in_bits: Optional[int] = field(default=4)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
        if type(self.target_modules) == str:
            self.target_modules = self.target_modules.split(',')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_on_inputs: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_files: Optional[List[str]] = field(default=None,
                                             metadata={"help": "The input training data file (a text file)."})
    validation_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_files is None and self.validation_files is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path): os.remove(pytorch_model_path)
            return control


def main():
    # 翻译：在src/transformers/training_args.py中查看所有可能的参数，或通过将--help标志传递给此脚本。现在我们保留了不同的参数集，以便更清晰地分离关注点。
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # pdb.set_trace() # 调试
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 翻译：如果我们只向脚本传递一个参数，并且它是json文件的路径，那么让我们解析它以获取我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # 命令行参数解析

    # 发送遥测。跟踪示例用法有助于我们更好地分配资源来维护它们。发送的信息是与您的Python/PyTorch版本一起传递的信息。
    send_example_telemetry("run_clm", model_args, data_args)

    # 设置日志记录。我们将日志记录到终端和日志文件中。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # 翻译：training_args.log_level的默认值是被动的，所以我们在这里将日志级别设置为info，以便有默认值。
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 翻译：在每个进程上记录小摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 翻译：检测最后一个检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 翻译：设置种子以确保在不同的进程中初始化相同的模型。
    set_seed(training_args.seed)

    # 翻译：获取数据集：您可以提供自己的CSV/JSON/TXT训练和评估文件（见下文），也可以只提供hub上可用的公共数据集之一的名称https://huggingface.co/datasets/（数据集将从数据集中心自动下载）。
    # 翻译：对于CSV/JSON文件，此脚本将使用名为"text"的列或如果没有名为"text"的列，则使用第一列。可以轻松地调整此行为（见下文）。
    # 翻译：在分布式训练中，load_dataset函数保证只有一个本地进程可以同时下载数据集。
    if True:
        data_files = {} # 数据文件
        dataset_args = {} # 数据集参数
        if data_args.train_files is not None: # 如果训练文件不为空
            data_files["train"] = data_args.train_files # 训练文件
        if data_args.validation_files is not None: # 如果验证文件不为空
            data_files["validation"] = data_args.validation_files # 验证文件
        extension = ( # 扩展名
            data_args.train_files[0].split(".")[-1] # 训练文件扩展名
            if data_args.train_files is not None # 如果训练文件不为空
            else data_args.validation_files.split(".")[-1] # 否则验证文件扩展名
        )
        if extension == "txt": # 如果扩展名为txt
            extension = "text" # 扩展名为text
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks # 是否保留换行符
        raw_datasets = load_dataset( # 加载数据集
            extension, # 扩展名
            data_files=data_files, # 数据文件
            cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'), # 缓存目录
            use_auth_token=True if model_args.use_auth_token else None, # 是否使用token
            **dataset_args, # 数据集参数
        )
        # 翻译：如果没有验证数据，validation_split_percentage将用于划分数据集。
        if "validation" not in raw_datasets.keys(): # 如果验证不在原始数据集键中
            raw_datasets["validation"] = load_dataset( # 加载数据集
                extension, # 扩展名
                data_files=data_files, # 数据文件
                split=f"train[:{data_args.validation_split_percentage}%]", # 划分
                cache_dir=model_args.cache_dir, # 缓存目录
                use_auth_token=True if model_args.use_auth_token else None, # 是否使用token
                **dataset_args, # 数据集参数
            )
            raw_datasets["train"] = load_dataset( # 加载数据集
                extension, # 扩展名
                data_files=data_files, # 数据文件
                split=f"train[{data_args.validation_split_percentage}%:]", # 划分
                cache_dir=model_args.cache_dir, # 缓存目录
                use_auth_token=True if model_args.use_auth_token else None, # 是否使用token
                **dataset_args, # 数据集参数
            )

    # 翻译：有关加载任何类型的标准或自定义数据集（从文件、python dict、pandas DataFrame等）的更多信息，请参见https://huggingface.co/docs/datasets/loading_datasets.html。
    # 翻译：加载预训练模型和tokenizer
    # 翻译：分布式训练：
    # 翻译：.from_pretrained方法保证只有一个本地进程可以同时下载模型和词汇表。
    config_kwargs = {
        "cache_dir": model_args.cache_dir,  # 缓存目录
        "revision": model_args.model_revision,  # 模型版本
        "use_auth_token": True if model_args.use_auth_token else None,  # 是否使用token
    }
    if model_args.config_name:  # 配置文件名
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)  # 从预训练模型中加载配置
    elif model_args.model_name_or_path:  # 模型名或路径
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)  # 从预训练模型中加载配置
    else:  # 从头开始训练
        config = CONFIG_MAPPING[model_args.model_type]()  # 从配置映射中加载配置
        logger.warning("You are instantiating a new config instance from scratch.")  # 警告：您正在从头开始实例化一个新的配置实例。
        if model_args.config_overrides is not None:  # 配置覆盖
            logger.info(f"Overriding config: {model_args.config_overrides}")  # 覆盖配置
            config.update_from_string(model_args.config_overrides)  # 从字符串更新配置
            logger.info(f"New config: {config}")  # 新配置

    tokenizer_kwargs = {  # tokenizer参数
        "cache_dir": model_args.cache_dir,  # 缓存目录
        "use_fast": model_args.use_fast_tokenizer,  # 是否使用快速tokenizer
        "revision": model_args.model_revision,  # 模型版本
        "use_auth_token": True if model_args.use_auth_token else None,  # 是否使用token
        "padding_side": 'left'  # 填充方向
    }
    if model_args.tokenizer_name:  # tokenizer名
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)  # 从预训练模型中加载tokenizer
    elif model_args.model_name_or_path:  # 模型名或路径
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  **tokenizer_kwargs)  # 从预训练模型中加载tokenizer
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."  # 正在从头开始实例化一个新的tokenizer。这个脚本不支持。
            "You can do it from another script, save it, and load it from here, using --tokenizer_name." # 可以从另一个脚本中执行此操作，保存它，然后使用--tokenizer_name从这里加载它。
        )
    tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token替换pad_token
    lora_config = LoraConfig(  # lora配置
        r = model_args.lora_r,  # r表示秩
        lora_alpha = model_args.lora_alpha,  # alpha表示缩放因子
        # target_modules = ["query_key_value"], # 目标模块
        # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'], # 目标模块
        target_modules = model_args.target_modules,  # 目标模块
        fan_in_fan_out = False,  # 是否使用fan_in_fan_out
        lora_dropout = 0.05,  # lora_dropout
        inference_mode = False,  # 是否使用推理模式
        bias = "none",  # 偏置
        task_type = "CAUSAL_LM",  # 任务类型
    )
    print(lora_config)
    bnb_config = BitsAndBytesConfig(  # bnb配置
        load_in_4bit=True,  # 是否使用4bit
        bnb_4bit_use_double_quant=True,  # 是否使用双量化
        bnb_4bit_quant_type="nf4",  # 量化类型
        bnb_4bit_compute_dtype=torch.bfloat16  # 计算类型
    )
    if model_args.model_name_or_path: # 模型名或路径
        torch_dtype = ( # torch数据类型
            model_args.torch_dtype # torch数据类型
            if model_args.torch_dtype in ["auto", None] # 如果torch数据类型在["auto", None]中
            else getattr(torch, model_args.torch_dtype) # 获取torch数据类型
        )
        print(torch_dtype)
        torch_dtype = torch.float16 # torch数据类型
        model = AutoModelForCausalLM.from_pretrained( # 从预训练模型中加载模型
            model_args.model_name_or_path, # 模型名或路径
            from_tf = bool(".ckpt" in model_args.model_name_or_path), # 是否从tensorflow加载
            config = config, # 配置
            cache_dir = model_args.cache_dir, # 缓存目录
            revision = model_args.model_revision, # 模型版本
            use_auth_token = True if model_args.use_auth_token else None, # 是否使用token
            torch_dtype = torch_dtype, # torch数据类型
            load_in_8bit = True if model_args.load_in_bits == 8 else False, # 是否使用8bit
            quantization_config = bnb_config if model_args.load_in_bits == 4 else None, # 量化配置
            # device_map  = 'auto'
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # 设备映射
        )
        # model = prepare_model_for_int8_training(model, output_embedding_layer_name="embed_out", layer_norm_names=[])

    else:
        model = AutoModelForCausalLM.from_config(config) # 从配置中加载模型
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values()) # 参数数量
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params") # 训练新模型

    # 翻译：我们只在必要时调整嵌入大小，以避免索引错误。如果您要从头开始创建一个小词汇表的模型，并且想要一个更小的嵌入大小，请删除此演示。
    embedding_size = model.get_input_embeddings().weight.shape[0] # 嵌入大小
    if len(tokenizer) > embedding_size: # 如果tokenizer长度大于嵌入大小
        model.resize_token_embeddings(len(tokenizer)) # 调整token嵌入
    if model_args.load_in_bits == 8: # 如果使用8bit
        model = prepare_model_for_int8_training(model) # 准备模型进行int8训练
    elif model_args.load_in_bits == 4: # 如果使用4bit
        model = prepare_model_for_kbit_training(model) # 准备模型进行kbit训练

    # 翻译：首先，我们将所有文本tokenize
    if training_args.do_train: # 如果训练
        column_names = list(raw_datasets["train"].features) # 列名
    else: # 否则
        column_names = list(raw_datasets["validation"].features) # 列名

    train_on_inputs = True # 训练输入
    if len(column_names) == 1: # 如果列数为1
        text_column_name = "text" if "text" in column_names else column_names[0] # 文本列名
    elif len(column_names) == 2: # 如果列数为2
        input_column_name = 'input' if 'input' in column_names else column_names[0] # 输入列名
        target_column_name = 'target' if 'target' in column_names else column_names[0] # 目标列名
        train_on_inputs = False # 训练输入
    else:
        raise ValueError('输入文件列数不对') # 输入文件列数不对
    print('train_on_inputs', train_on_inputs) # 训练输入
    # 翻译：由于这将被pickled以避免哈希器中的_LazyModule错误，强制logger在tokenize_function之前加载
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples): # tokenize函数
        with CaptureLogger(tok_logger) as cl: # 捕获日志
            output = tokenizer([item for item in examples[text_column_name]], truncation=True, max_length=data_args.block_size, padding=False, return_tensors=None) # tokenizer
            output['labels'] = output['input_ids'].copy() # 标签
        return output # 输出

    def tokenize(prompt): # tokenize函数
        result = tokenizer(prompt, truncation=True, max_length=data_args.block_size, padding=False, return_tensors=None) # tokenizer
        result["labels"] = result["input_ids"].copy() # 标签
        return result # 结果

    def generate_and_tokenize_prompt(data_point): # 生成和tokenize提示
        input_text = data_point[input_column_name] # 输入文本
        target_text = data_point[target_column_name] # 目标文本
        full_prompt = input_text + target_text # 完整提示
        tokenized_full_prompt = tokenize(full_prompt) # tokenize完整提示
        if not train_on_inputs: # 如果不是训练输入
            user_prompt = input_text # 用户提示
            tokenized_user_prompt = tokenize(user_prompt) # tokenize用户提示
            user_prompt_len = len(tokenized_user_prompt["input_ids"]) # 用户提示长度
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:] # 标签
        return tokenized_full_prompt # tokenize完整提示

    with training_args.main_process_first(desc="dataset map tokenization"): # 主进程
        if not data_args.streaming: # 如果不是流式
            tokenized_datasets = raw_datasets.map(  # tokenize数据集
                tokenize_function if train_on_inputs == True else generate_and_tokenize_prompt, # tokenize函数
                batched=True if train_on_inputs == True else False, # 批处理
                num_proc=data_args.preprocessing_num_workers, # 预处理进程数
                remove_columns=column_names, # 删除列
                load_from_cache_file=not data_args.overwrite_cache, # 从缓存文件加载
                desc="Running tokenizer on dataset", # 运行tokenizer
            )
        else: # 如果是流式
            tokenized_datasets = raw_datasets.map( # tokenize数据集
                tokenize_function if train_on_inputs == True else generate_and_tokenize_prompt, # tokenize函数
                batched=True if train_on_inputs == True else False, # 批处理
                remove_columns=column_names, # 删除列
            )

    if data_args.block_size is None: # 如果块大小为None
        block_size = tokenizer.model_max_length # 块大小
        if block_size > 2048: # 如果块大小大于2048
            block_size = 2048 # 块大小
    else: # 否则
        block_size = min(data_args.block_size, tokenizer.model_max_length) # 块大小

    if training_args.do_train: # 如果训练
        if "train" not in tokenized_datasets: # 如果训练集不在tokenized_datasets中
            raise ValueError("--do_train requires a train dataset") # --do_train需要一个训练数据集
        train_dataset = tokenized_datasets["train"] # 训练数据集
        if data_args.max_train_samples is not None: # 如果最大训练样本不为None
            max_train_samples = min(len(train_dataset), data_args.max_train_samples) # 最大训练样本
            train_dataset = train_dataset.select(range(max_train_samples)) # 选择
        for index in random.sample(range(len(train_dataset)), 3): # 随机选择
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.") # 训练集样本
        train_dataset = train_dataset.shuffle(seed=training_args.seed) # 打乱

    if training_args.do_eval: # 如果评估
        if "validation" not in tokenized_datasets: # 如果验证集不在tokenized_datasets中
            raise ValueError("--do_eval requires a validation dataset") # --do_eval需要一个验证数据集
        eval_dataset = tokenized_datasets["validation"] # 验证数据集
        if data_args.max_eval_samples is not None: # 如果最大评估样本不为None
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples) # 最大评估样本
            eval_dataset = eval_dataset.select(range(max_eval_samples)) # 选择

        def preprocess_logits_for_metrics(logits, labels): # 为指标预处理logits
            if isinstance(logits, tuple): # 如果logits是元组
                # 翻译：根据模型和配置，logits可能包含额外的张量，如past_key_values，但logits始终首先出现
                logits = logits[0] # logits
            return logits.argmax(dim=-1) # 返回logits的最大值

        metric = evaluate.load("accuracy.py") # 加载指标accuracy.py

        def compute_metrics(eval_preds): # 计算指标
            preds, labels = eval_preds # 预测，标签
            # 翻译：预测在预处理logits_for_metrics计算argmax(-1)之后与标签具有相同的形状，但我们需要shift标签
            labels = labels[:, 1:].reshape(-1)
            # .reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            # .reshape(-1)
            # print(labels.shape)
            # true_predictions = [
            #     [p for (p, l) in zip(pred, gold_label) if l != -100]
            #     for pred, gold_label in zip(preds, labels)
            # ]
            # true_labels = [
            #     [l for (p, l) in zip(pred, gold_label) if l != -100]
            #     for pred, gold_label in zip(preds, labels)
            # ]            
            # preds = np.array(true_predictions).reshape(-1)
            # labels = np.array(true_labels).reshape(-1)
            return metric.compute(predictions=preds, references=labels)
        # layer_norm_names=[]

    model = get_peft_model(model, lora_config) # 获取peft模型
    model.print_trainable_parameters() # 打印可训练参数

    # 初始化我们的Trainer
    trainer = Trainer( # 训练器
        model = model, # 模型
        args = training_args, # 训练参数
        train_dataset = train_dataset if training_args.do_train else None, # 训练数据集
        eval_dataset = eval_dataset if training_args.do_eval else None, # 评估数据集
        tokenizer = tokenizer, # tokenizer
        # 数据收集器将默认为DataCollatorWithPadding，因此我们将其更改
        data_collator = transformers.DataCollatorForSeq2Seq( # 数据收集器
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True # tokenizer，填充到8的倍数，返回张量，填充
        ),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None, # 计算指标
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None, # 为指标预处理logits
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None), # 回调
    )

    # 训练
    if training_args.do_train: # 如果训练
        checkpoint = None # 检查点
        if training_args.resume_from_checkpoint is not None: # 如果从检查点恢复
            resume_from_checkpoint = training_args.resume_from_checkpoint # 从检查点恢复
            checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin") # 检查点名
            if not os.path.exists(checkpoint_name): # 如果检查点名不存在
                checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin") # 翻译：只有LoRA模型-上面的LoRA配置必须适合
                resume_from_checkpoint = (False) # 因此训练器不会尝试加载其状态
            # 翻译：上面的两个文件根据它们是如何保存的而有不同的名称，但实际上是相同的
            if os.path.exists(checkpoint_name): # 如果检查点名存在
                print(f"Restarting from {checkpoint_name}") # 从检查点名重启
                adapters_weights = torch.load(checkpoint_name) # 适配器权重
                set_peft_model_state_dict(model, adapters_weights) # 设置peft模型状态字典
            else: # 否则
                print(f"Checkpoint {checkpoint_name} not found") # 检查点未找到
            # checkpoint = Fa
        elif last_checkpoint is not None: # 如果最后一个检查点不为None
            checkpoint = last_checkpoint # 检查点

        if torch.__version__ >= "2" and sys.platform != "win32": # 如果torch版本大于等于2且系统平台不是win32
            model = torch.compile(model) # 编译模型

        train_result = trainer.train(resume_from_checkpoint=checkpoint) # 训练结果
        trainer.save_model() # 保存tokenizer对于简单的上传

        metrics = train_result.metrics # 指标

        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)) # 最大训练样本
        metrics["train_samples"] = min(max_train_samples, len(train_dataset)) # 训练样本

        trainer.log_metrics("train", metrics) # 训练指标
        trainer.save_metrics("train", metrics) # 保存训练指标
        trainer.save_state() # 保存状态

    # Evaluation
    if training_args.do_eval: # 如果评估
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate() # 评估

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset) # 最大评估样本
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset)) # 评估样本
        try:
            perplexity = math.exp(metrics["eval_loss"]) # 困惑度
        except OverflowError:
            perplexity = float("inf") # 无穷大
        metrics["perplexity"] = perplexity # 困惑度

        trainer.log_metrics("eval", metrics) # 评估指标
        trainer.save_metrics("eval", metrics) # 保存评估指标


def _mp_fn(index):
    # 对于xla_spawn（TPUs）
    main()


if __name__ == "__main__":
    main()