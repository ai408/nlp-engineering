#!/usr/bin/env python
# coding=utf-8
"""
翻译：在文本文件或数据集上对语言建模库模型进行微调（GPT、GPT-2、CTRL等）。这里是可以通过这个脚本进行微调的hub上的检查点的完整列表：
https://huggingface.co/models?filter=text-generation
"""
# 翻译：您还可以在自己的因果语言建模任务中调整此脚本。这方面的说明留在了注释中。

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from itertools import chain
# import deepspeed
from typing import Optional,List

import datasets
import pandas as pd
import evaluate
import torch
from datasets import load_dataset
from datasets.combine import interleave_datasets
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
    is_torch_tpu_available,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import interleave_datasets

# Will error if the minimal version of Transformers is not installed. Remove at your own risks. check_min_version("4.27.0.dev0")
# 翻译：如果没有安装Transformers的最小版本，将会出错。自行决定是否删除。check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass # 数据类
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    # 与我们将要微调或从头开始训练的model/config/tokenizer有关的参数。
    """

    model_name_or_path: Optional[str] = field( # 模型名称或路径
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch." # 用于权重初始化的模型检查点。如果要从头开始训练模型，请不要设置。
            )
        },
    )
    model_type: Optional[str] = field( # 模型类型
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)}, # 如果从头开始训练，请从列表中传递一个模型类型
    )
    config_overrides: Optional[str] = field( # 配置覆盖
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: " # 当从头开始训练模型时，覆盖一些现有的默认配置设置。例子：
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"                      # n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index
            )
        },
    )
    config_name: Optional[str] = field( # 配置名称
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"} # 预训练配置名称或路径（如果与model_name不同）
    )
    tokenizer_name: Optional[str] = field( # 分词器名称
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"} # 预训练的分词器名称或路径（如果与model_name不同）
    )
    cache_dir: Optional[str] = field( # 缓存目录
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, # 您想从huggingface.co下载的预训练模型存储在哪里
    )
    use_fast_tokenizer: bool = field( # 使用快速分词器
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, # 是否使用快速分词器（由tokenizers库支持）
    )
    model_revision: str = field( # 模型修订
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}, # 要使用的特定模型版本（可以是分支名称、标签名称或提交ID）。
    )
    use_auth_token: bool = field( # 使用授权令牌
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models)." # 将使用在运行`huggingface-cli login`时生成的令牌（使用私有模型运行此脚本时必需）。
            )
        },
    )
    torch_dtype: Optional[str] = field( # torch数据类型
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights." # 覆盖默认的`torch.dtype`并在此dtype下加载模型。如果传递了`auto`，则dtype将自动从模型的权重中派生。
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"], # 选择：auto，bfloat16，float16，float32
        },
    )

    def __post_init__(self): # 后初始化
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None): # 如果配置覆盖不是空的并且（配置名称不是空的或者模型名称或路径不是空的）
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path" # --config_overrides不能与--config_name或--model_name_or_path组合使用
            )


@dataclass # 数据类
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    # 与我们将要输入模型进行训练和评估的数据有关的参数。
    """

    dataset_name: Optional[str] = field( # 数据集名称
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."} # 要使用的数据集的名称（通过数据集库）。
    )
    dataset_config_name: Optional[str] = field( # 数据集配置名称
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."} # 要使用的数据集的配置名称（通过数据集库）。
    )
    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."}) # 输入训练数据文件（文本文件）。
    validation_files: Optional[List[str]]  = field( # 验证文件
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}, # 一个可选的输入评估数据文件，用于评估困惑度（文本文件）。
    )
    max_train_samples: Optional[int] = field( # 最大训练样本
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set." # 为了调试目的或更快的训练，如果设置了这个值，就将训练示例的数量截断为这个值。
            )
        },
    )
    max_eval_samples: Optional[int] = field( # 最大评估样本
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set." # 为了调试目的或更快的训练，如果设置了这个值，就将评估示例的数量截断为这个值。
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"}) # 启用流模式
    block_size: Optional[int] = field( # 块大小
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. " # 分词后的可选输入序列长度。
                "The training dataset will be truncated in block of this size for training. " # 训练数据集将被截断为这个大小的块进行训练。
                "Default to the model max input length for single sentence inputs (take into account special tokens)." # 默认为单句输入的模型最大输入长度（考虑特殊标记）。
            )
        },
    )
    overwrite_cache: bool = field( # 覆盖缓存
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"} # 覆盖缓存的训练和评估集
    )
    validation_split_percentage: Optional[int] = field( # 验证分割百分比
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split" # 如果没有验证分割，训练集中用作验证集的百分比
        },
    )
    preprocessing_num_workers: Optional[int] = field( # 预处理工作进程数
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}, # 用于预处理的进程数。
    )
    keep_linebreaks: bool = field( # 保留换行符
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."} # 在使用TXT文件时是否保留换行符。
    )

    def __post_init__(self): # 后初始化
        if self.streaming: # 如果启用流模式
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`") # 流功能需要`datasets>=2.0.0`

        if self.dataset_name is None and self.train_files is None and self.validation_files is None: # 如果数据集名称为空并且训练文件为空并且验证文件为空
            raise ValueError("Need either a dataset name or a training/validation file.") # 需要数据集名称或训练/验证文件。
        else:
            if self.train_files is not None: # 如果训练文件不为空
                extension = self.train_files[0].split(".")[-1] # 扩展名
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."   # `train_file`应该是一个csv、json或txt文件。
            if self.validation_files is not None: # 如果验证文件不为空
                extension = self.validation_files[0].split(".")[-1] # 扩展名
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file." # `validation_file`应该是一个csv、json或txt文件。
                
def main():
    # 翻译：请参阅src/transformers/training_args.py中的所有可能参数，或通过将--help标志传递给此脚本。
    # 翻译：我们现在保留不同的参数集，以便更清晰地分离关注点。
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)) # 三个参数类，分别是模型参数，数据参数，训练参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 翻译：如果我们只向脚本传递一个参数，并且它是json文件的路径，那么让我们解析它以获取我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The information sent is the one passed as arguments along with your Python/PyTorch versions.
    # 翻译：发送遥测。跟踪示例用法有助于我们更好地分配资源来维护它们。发送的信息是与Python/PyTorch版本一起传递的信息。
    send_example_telemetry("run_clm", model_args, data_args)

    # 设置日志记录
    logging.basicConfig( # 基本配置
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # 格式
        datefmt="%m/%d/%Y %H:%M:%S", # 日期格式
        handlers=[logging.StreamHandler(sys.stdout)], # 处理程序
    )

    if training_args.should_log: # 如果应该记录
        # 翻译：training_args.log_level的默认值是被动的，所以我们在这里将日志级别设置为info，以便有默认值。
        transformers.utils.logging.set_verbosity_info() # 设置详细程度信息

    log_level = training_args.get_process_log_level() # 获取进程日志级别
    logger.setLevel(log_level) # 设置日志级别
    datasets.utils.logging.set_verbosity(log_level) # 设置详细程度
    transformers.utils.logging.set_verbosity(log_level) # 设置详细程度
    transformers.utils.logging.enable_default_handler() # 启用默认处理程序
    transformers.utils.logging.enable_explicit_format() # 启用显式格式

    # 翻译：在每个进程上记录小摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}") # 训练/评估参数

    # 翻译：检测最后一个检查点。
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir: # 如果输出目录是目录并且训练并且不覆盖输出目录
        last_checkpoint = get_last_checkpoint(training_args.output_dir) # 获取最后一个检查点
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0: # 如果最后一个检查点为空并且输出目录的文件数大于0
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. " # 输出目录（{training_args.output_dir}）已经存在并且不为空。
                "Use --overwrite_output_dir to overcome." # 使用--overwrite_output_dir来克服。
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None: # 如果最后一个检查点不为空并且不是从检查点恢复
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change " # 检测到检查点，在{last_checkpoint}处恢复训练。为了避免这种行为，改变
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch." # `--output_dir`或添加`--overwrite_output_dir`从头开始训练。
            )

    # 翻译：设置种子以初始化模型。
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below) or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the datasets Hub).
    # 翻译：获取数据集：您可以提供自己的CSV/JSON/TXT训练和评估文件（见下文），也可以只提供hub上可用的公共数据集之一的名称，在https://huggingface.co/datasets/（数据集将从数据集中心自动下载）。
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called 'text' is found. You can easily tweak this behavior (see below).
    # 翻译：对于CSV/JSON文件，此脚本将使用名为“text”的列或如果没有名为“text”的列，则使用第一列。您可以轻松地调整这种行为（见下文）。
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently download the dataset.
    # 翻译：在分布式训练中，load_dataset函数保证只有一个本地进程可以同时下载数据集。
    if True:
        data_files = {} # 数据文件
        dataset_args = {} # 数据集参数
        if data_args.train_files is not None: # 如果训练文件不为空

            print(data_args.train_files) # 打印训练文件
            data_files["train"] = data_args.train_files # 训练文件
            print('训练文件总个数',len(data_args.train_files)) # 训练文件总个数
        if data_args.validation_files is not None: # 如果验证文件不为空
            data_files["validation"] = data_args.validation_files # 验证文件
        extension = ( # 扩展名
            data_files["train"][0].split(".")[-1]
            if data_files["train"] is not None
            else data_args.validation_files.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        
        
        raw_datasets = load_dataset( # 加载数据集
            extension, # 扩展名
            data_files=data_files, # 数据文件
            streaming=data_args.streaming, # 流
            cache_dir=os.path.join(training_args.output_dir,'dataset_cache'), # 缓存目录
            use_auth_token=True if model_args.use_auth_token else None, # 使用授权令牌
            **dataset_args, # 数据集参数
        )
        if data_args.streaming: # 如果启用流
            raw_datasets = raw_datasets.shuffle(seed=training_args.seed, buffer_size=1000000) # 打乱
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        # 翻译：如果没有验证数据，将使用validation_split_percentage来划分数据集。
        if "validation" not in raw_datasets.keys(): # 如果验证不在原始数据集的键中
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at https://huggingface.co/docs/datasets/loading_datasets.html.
    # 翻译：有关加载任何类型的标准或自定义数据集（从文件、python dict、pandas DataFrame等）的更多信息，请参见https://huggingface.co/docs/datasets/loading_datasets.html。

    # Load pretrained model and tokenizer
    # 翻译：加载预训练模型和分词器
    # Distributed training: The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    # 翻译：分布式训练：.from_pretrained方法保证只有一个本地进程可以同时下载模型和词汇表。
    config_kwargs = {
        "cache_dir": model_args.cache_dir, # 缓存目录
        "revision": model_args.model_revision, # 模型修订
        "use_auth_token": True if model_args.use_auth_token else None, # 使用授权令牌
    }
    if model_args.config_name: # 如果配置名称
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs) # 从预训练中加载配置
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs) # 从预训练中加载配置
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.") # 您正在从头开始实例化一个新的配置实例。
        if model_args.config_overrides is not None: # 如果配置覆盖不为空
            logger.info(f"Overriding config: {model_args.config_overrides}") # 覆盖配置
            config.update_from_string(model_args.config_overrides) # 从字符串更新
            logger.info(f"New config: {config}") # 新配置

    print(training_args.local_rank,'start load tokenizer') # 开始加载分词器
    tokenizer_kwargs = { # 分词器参数
        "cache_dir": model_args.cache_dir, # 缓存目录
        "use_fast": model_args.use_fast_tokenizer, # 使用快速分词器
        "revision": model_args.model_revision, # 模型修订
        "use_auth_token": True if model_args.use_auth_token else None, # 使用授权令牌
    }
    if model_args.tokenizer_name: # 如果分词器名称
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs) # 从预训练中加载分词器
    elif model_args.model_name_or_path: # 如果模型名称或路径
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs) # 从预训练中加载分词器
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script." # 您正在从头开始实例化一个新的分词器。这个脚本不支持。
            "You can do it from another script, save it, and load it from here, using --tokenizer_name." # 您可以从另一个脚本中执行它，保存它，并使用--tokenizer_name从这里加载它。
        )
    print(training_args.local_rank,'end load tokenizer') # 结束加载分词器
    print(training_args.local_rank,'start load model') # 开始加载模型
    if model_args.model_name_or_path: # 如果模型名称或路径
        torch_dtype = ( # torch数据类型
            model_args.torch_dtype # torch数据类型
            if model_args.torch_dtype in ["auto", None] # 如果torch数据类型在auto或None中
            else getattr(torch, model_args.torch_dtype) # 获取torch数据类型
        )
        model = AutoModelForCausalLM.from_pretrained( # 从预训练中加载因果语言模型
            model_args.model_name_or_path, # 模型名称或路径
            from_tf=bool(".ckpt" in model_args.model_name_or_path), # 来自tf
            config=config, # 配置
            cache_dir=model_args.cache_dir, # 缓存目录
            revision=model_args.model_revision, # 模型修订
            use_auth_token=True if model_args.use_auth_token else None, # 使用授权令牌
        )
    else:
        model = AutoModelForCausalLM.from_config(config) # 从配置中加载因果语言模型
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values()) # 参数数量
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params") # 从头开始训练新模型-总大小={n_params/2**20:.2f}M params
    print(training_args.local_rank,'end load model') # 结束加载模型
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch on a small vocab and want a smaller embedding size, remove this demo.
    # 翻译：我们只在必要时调整嵌入大小，以避免索引错误。如果您从头开始创建一个小词汇表的模型，并且想要一个更小的嵌入大小，请删除此演示。
    embedding_size = model.get_input_embeddings().weight.shape[0] # 嵌入大小
    if len(tokenizer) > embedding_size: # 如果分词器长度大于嵌入大小
        model.resize_token_embeddings(len(tokenizer)) # 调整嵌入大小
    # Preprocessing the datasets. First we tokenize all the texts.
    # 翻译：预处理数据集。首先我们对所有的文本进行分词。
    if training_args.do_train: # 如果训练
        if data_args.streaming: # 如果启用流
            dataset_head = raw_datasets["train"].take(3) # 取前三个
            print(list(dataset_head)) # 打印
            column_names = list(list(dataset_head)[0].keys()) # 列名
        else: # 否则
            column_names = list(raw_datasets["train"].features) # 列名
    else: # 否则
        if data_args.streaming: # 如果启用流
            dataset_head = raw_datasets["validation"].take(3) # 取前三个
            column_names = list(list(dataset_head)[0].keys()) # 列名
        else: # 否则
            column_names = list(raw_datasets["validation"].features) # 列名
    print(column_names) # 打印列名
    text_column_name = "text" if "text" in column_names else column_names[0] # 文本列名

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # 翻译：由于这将被pickled以避免哈希器中的_LazyModule错误，在tokenize_function之前强制logger加载
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base") # 获取日志记录器

    def tokenize_function(examples): # 分词函数
        with CaptureLogger(tok_logger) as cl: # 捕获日志记录器
            output = tokenizer( [ '<s>'+item+'</s>' for item in examples[text_column_name]]) # 分词
        return output # 输出

    with training_args.main_process_first(desc="dataset map tokenization"): # 数据集映射分词
        if not data_args.streaming: # 如果不启用流
            tokenized_datasets = raw_datasets.map( # 映射
                tokenize_function, # 分词函数
                batched=True, # 批处理
                num_proc=data_args.preprocessing_num_workers, # 预处理工作进程数
                remove_columns=column_names, # 删除列
                load_from_cache_file=not data_args.overwrite_cache, # 从缓存文件加载
                desc="Running tokenizer on dataset", # 在数据集上运行分词器
            )
        else: # 否则
            tokenized_datasets = raw_datasets.map( # 映射
                tokenize_function, # 分词函数
                batched=True, # 批处理
                remove_columns=column_names, # 删除列
                batch_size = 60000, # 批大小
            )

    if data_args.block_size is None: # 如果块大小为空
        block_size = tokenizer.model_max_length # 块大小
        if block_size > 1024: # 如果块大小大于1024
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can override this default with `--block_size xxx`."
                # 所选的分词器支持比默认的`block_size`值1024更长的`model_max_length`。如果您想使用更长的`block_size`，最多可以使用`tokenizer.model_max_length`，您可以使用`--block_size xxx`覆盖此默认值。
            )
            block_size = 1024 # 块大小
    else:
        if data_args.block_size > tokenizer.model_max_length: # 如果块大小大于分词器模型最大长度
            logger.warning( # 警告
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model" # 传递的block_size（{data_args.block_size}）大于模型的最大长度
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}." # （{tokenizer.model_max_length}）。使用block_size={tokenizer.model_max_length}。
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length) # 块大小

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # 翻译：主数据处理函数，它将连接我们数据集中的所有文本并生成块大小的块。
    def group_texts(examples):
        # 翻译：连接所有文本。
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} # 连接所有文本
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) # 总长度
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        # 翻译：我们丢弃了小的余数，如果模型支持，我们可以添加填充，而不是这个丢弃，您可以根据需要自定义这部分。
        if total_length >= block_size: # 如果总长度大于块大小
            total_length = (total_length // block_size) * block_size # 总长度
        # Split by chunks of max_len.
        # 翻译：按max_len分割。
        result = { # 结果
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)] # 拼接的示例
            for k, t in concatenated_examples.items() # 拼接的示例
        }
        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
        logger.info("group texts input examples length%d after_group size%d"%(len(examples['input_ids']),len(result["input_ids"]))) # 组文本输入示例长度%d后组大小%d
        result["labels"] = result["input_ids"].copy() # 标签
        return result # 返回结果

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower to preprocess.
    # 翻译：请注意，使用`batched=True`，此映射一次处理1,000个文本，因此group_texts为这些1,000个文本组中的每一个丢弃了一个余数。您可以在这里调整batch_size，但较高的值可能会更慢地预处理。
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # 翻译：为了加快这一部分的速度，我们使用多处理。有关map方法的更多信息，请参见文档：
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"): # 将文本分组在一起
        if not data_args.streaming: # 如果不启用流
            lm_datasets = tokenized_datasets.map( # 映射
                group_texts, # 分组文本
                batched=True, # 批处理
                num_proc=data_args.preprocessing_num_workers, # 预处理工作进程数
                load_from_cache_file=not data_args.overwrite_cache, # 从缓存文件加载
                desc=f"Grouping texts in chunks of {block_size}", # 将文本分组在{block_size}的块中
                batch_size = 40000, # 批大小
            )
        else: # 否则
            lm_datasets = tokenized_datasets.map( # 映射
                group_texts, # 分组文本
                batched=True, # 批处理
                batch_size = 60000, # 批大小
            )
    print(training_args.local_rank,'start select train_dataset') # 开始选择训练数据集
    if training_args.do_train: # 如果训练
        if "train" not in tokenized_datasets: # 如果训练不在分词数据集中
            raise ValueError("--do_train requires a train dataset") # --do_train需要一个训练数据集
        train_dataset = lm_datasets["train"] # 训练数据集
        if data_args.max_train_samples is not None and data_args.streaming==False: # 如果最大训练样本不为空并且不是流
            max_train_samples = min(len(train_dataset), data_args.max_train_samples) # 最大训练样本
            train_dataset = train_dataset.select(range(max_train_samples)) # 选择
    print(training_args.local_rank,'end select train_dataset') # 结束选择训练数据集

    if training_args.do_eval: # 如果评估
        if "validation" not in tokenized_datasets: # 如果验证不在分词数据集中
            raise ValueError("--do_eval requires a validation dataset") # --do_eval需要一个验证数据集
        print(training_args.local_rank,'start select eval_dataset') # 开始选择评估数据集
        eval_dataset = lm_datasets["validation"] # 评估数据集
        if data_args.max_eval_samples is not None and data_args.streaming==False : # 如果最大评估样本不为空并且不是流
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples) # 最大评估样本
            eval_dataset = eval_dataset.select(range(max_eval_samples)) # 选择
        print(training_args.local_rank,'end select eval_dataset') # 结束选择评估数据集
        def preprocess_logits_for_metrics(logits, labels): # 为指标预处理logits
            if isinstance(logits, tuple): # 如果logits是元组
                # Depending on the model and config, logits may contain extra tensors, like past_key_values, but logits always come first
                # 翻译：根据模型和配置，logits可能包含额外的张量，比如past_key_values，但logits总是首先出现的。
                logits = logits[0] # logits
            return logits.argmax(dim=-1) # 返回argmax(-1)
        print(training_args.local_rank,'start load metric') # 开始加载指标
        metric = evaluate.load("accuracy.py") # 加载指标
        print(training_args.local_rank,'end load metric') # 结束加载指标

        def compute_metrics(eval_preds): # 计算指标
            preds, labels = eval_preds # 预测，标签
            # preds have the same shape as the labels, after the argmax(-1) has been calculated by preprocess_logits_for_metrics but we need to shift the labels
            # 翻译：在preprocess_logits_for_metrics计算argmax(-1)之后，preds与标签具有相同的形状，但我们需要移动标签。
            labels = labels[:, 1:].reshape(-1) # 标签
            preds = preds[:, :-1].reshape(-1) # 预测
            return metric.compute(predictions=preds, references=labels) # 计算指标
    
    print(training_args.local_rank,'Initialize our Trainer') # 初始化我们的训练器
    # training_args.device = "cpu"
    trainer = Trainer( # 训练器
        model=model, # 模型
        args=training_args, # 训练参数
        train_dataset= IterableWrapper(train_dataset) if training_args.do_train else None, # 训练数据集
        eval_dataset= IterableWrapper(eval_dataset) if training_args.do_eval else None, # 评估数据集
        tokenizer=tokenizer, # 分词器
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # 翻译：数据收集器将默认为DataCollatorWithPadding，因此我们将其更改。
        data_collator=default_data_collator, # 默认数据收集器
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None, # 计算指标
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None, # 为指标预处理logits
        # callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )
    
    if training_args.do_train: # 如果训练
        checkpoint = None # 检查点
        if training_args.resume_from_checkpoint is not None: # 如果从检查点恢复
            checkpoint = training_args.resume_from_checkpoint # 检查点
        elif last_checkpoint is not None: # 否则如果最后一个检查点不为空
            checkpoint = last_checkpoint # 检查点

        print(training_args.local_rank,'start train') # 开始训练
        train_result = trainer.train(resume_from_checkpoint=checkpoint) # 训练结果
        trainer.save_model()  # Saves the tokenizer too for easy upload # 也为了方便上传保存分词器

        metrics = train_result.metrics # 指标

        max_train_samples = ( # 最大训练样本
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset) # 如果最大训练样本不为空则为最大训练样本否则为训练数据集长度
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset)) # 训练样本

        trainer.log_metrics("train", metrics) # 训练指标
        trainer.save_metrics("train", metrics) # 保存指标
        trainer.save_state() # 保存状态

    # Evaluation
    if training_args.do_eval: # 如果评估
        logger.info("*** Evaluate ***") # 评估

        metrics = trainer.evaluate() # 评估

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset) # 最大评估样本
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset)) # 评估样本
        try:
            perplexity = math.exp(metrics["eval_loss"]) # 困惑
        except OverflowError:
            perplexity = float("inf") # 无穷大
        metrics["perplexity"] = perplexity # 困惑

        trainer.log_metrics("eval", metrics) # 评估指标
        trainer.save_metrics("eval", metrics) # 保存指标



def _mp_fn(index): # 多进程函数
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()