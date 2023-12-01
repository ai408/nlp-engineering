"""
# 翻译：在文本文件或数据集上对因果语言建模（GPT、GPT-2、CTRL等）的库模型进行微调。
# 翻译：这是可以通过此脚本进行微调的hub上的检查点的完整列表：https://huggingface.co/models?filter=text-generation
"""
# 翻译：还可以根据自己的因果语言建模任务调整此脚本。这方面的Pointers留在了评论中。
import logging
import math
import os
import sys
import random
from dataclasses import dataclass, field
from itertools import chain
# import deepspeed
from typing import Optional,List,Union

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


# 翻译：如果未安装transformers的最小版本，将出错。自行决定是否删除。check_min_version("4.27.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    翻译：与我们将要微调或从头开始训练的model/config/tokenizer有关的参数。
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
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
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

    # def __post_init__(self):
    #     if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
    #         raise ValueError(
    #             "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
    #         )
    #     if type(self.target_modules)==str:
    #         self.target_modules = self.target_modules.split(',')


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
    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_files: Optional[List[str]]  = field(
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

    # def __post_init__(self):
    #     if self.streaming:
    #         require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
    #
    #     if self.dataset_name is None and self.train_files is None and self.validation_files is None:
    #         raise ValueError("Need either a dataset name or a training/validation file.")
    #     else:
    #         if self.train_files is not None:
    #             extension = self.train_files[0].split(".")[-1]
    #             assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
    #         if self.validation_files is not None:
    #             extension = self.validation_files[0].split(".")[-1]
    #             assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def main():
    # 翻译：请参阅src/transformers/training_args.py中的所有可能参数，或通过将--help标志传递给此脚本。
    # 翻译：现在我们保留了不同的参数集，以便更清晰地分离关注点。
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)) # 参数解析器
    # pdb.set_trace() # 设置断点
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # 如果命令行参数只有一个，且是json文件
        # 翻译：如果我们只向脚本传递一个参数，并且它是json文件的路径，那么让我们解析它以获取我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])) # 从json文件中解析参数
    else: # 如果命令行参数不是json文件
        model_args, data_args, training_args = parser.parse_args_into_dataclasses() # 从命令行中解析参数

    # 翻译：发送遥测信息。跟踪示例用法有助于我们更好地分配资源来维护它们。发送的信息是与您的Python/PyTorch版本一起传递的信息。
    send_example_telemetry("run_clm", model_args, data_args) # 发送遥测信息

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 设置日志级别
    if training_args.should_log:
        # 翻译：training_args.log_level的默认值是被动的，所以我们在这里设置日志级别为info，以便有这个默认值。
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level() # 获取日志级别
    logger.setLevel(log_level) # 设置日志级别
    datasets.utils.logging.set_verbosity(log_level) # 设置日志级别
    transformers.utils.logging.set_verbosity(log_level) # 设置日志级别
    transformers.utils.logging.enable_default_handler() # 设置日志级别
    transformers.utils.logging.enable_explicit_format() # 设置日志级别

    # 在每个进程上记录小摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检测最后一个检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir: # 如果训练模型，且输出目录存在，且不覆盖输出目录
        last_checkpoint = get_last_checkpoint(training_args.output_dir) # 获取最后一个检查点
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0: # 如果最后一个检查点不存在，且输出目录不为空
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None: # 如果最后一个检查点存在，且不从检查点恢复
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 初始化模型前设置随机种子
    set_seed(training_args.seed)

    # 翻译：获取数据集：您可以提供自己的CSV/JSON/TXT训练和评估文件（见下文），也可以只提供公共数据集之一的名称，这些数据集可以在https://huggingface.co/datasets/（数据集将从datasets Hub自动下载）。
    # 翻译：对于CSV/JSON文件，此脚本将使用名为“text”的列或第一列（如果没有名为“text”的列）。
    # 翻译：在分布式训练中，load_dataset函数保证只有一个本地进程可以同时下载数据集。
    if True:
        data_files = {}
        dataset_args = {}
        if data_args.train_files is not None:
            data_files["train"] = data_args.train_files
        if data_args.validation_files is not None:
            data_files["validation"] = data_args.validation_files
        extension = (
            data_args.train_files[0].split(".")[-1]
            if data_args.train_files is not None
            else data_args.validation_files.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=os.path.join(training_args.output_dir,'dataset_cache'),
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # 翻译：如果没有验证数据，validation_split_percentage将用于划分数据集。
        if "validation" not in raw_datasets.keys():
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

    # 翻译：有关加载任何类型的标准或自定义数据集（从文件、python dict、pandas DataFrame等）的更多信息，请参见https://huggingface.co/docs/datasets/loading_datasets.html。
    # 翻译：加载预训练模型和标记器
    # 翻译：分布式训练：
    # 翻译：.from_pretrained方法保证只有一个本地进程可以同时下载模型和词汇表。
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,                               # 缓存目录
        "use_fast": model_args.use_fast_tokenizer,                       # 是否使用快速分词器
        "revision": model_args.model_revision,                           # 模型版本
        "use_auth_token": True if model_args.use_auth_token else None,   # 是否使用token
        "padding_side":'left'                                            # 填充方向
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        print(torch_dtype)
        torch_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,                                 # 模型名称或路径
            from_tf=bool(".ckpt" in model_args.model_name_or_path),        # 是否从tensorflow加载
            config=config,                                                 # 配置
            cache_dir=model_args.cache_dir,                                # 缓存目录
            revision=model_args.model_revision,                            # 模型版本
            use_auth_token=True if model_args.use_auth_token else None,    # 是否使用token
            torch_dtype=torch_dtype,                                       # torch数据类型
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}        # 设备映射
        )
        # model = prepare_model_for_int8_training(model, output_embedding_layer_name="embed_out", layer_norm_names=[])
        
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # 翻译：我们只在必要时调整嵌入大小，以避免索引错误。如果您从头开始创建一个小词汇表的模型，并且想要一个更小的嵌入大小，请删除此演示。
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # 翻译：对数据集进行预处理。首先我们对所有文本进行标记化。
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
        
    train_on_inputs = True
    if len(column_names)==1:
        text_column_name = "text" if "text" in column_names else column_names[0]
    elif len(column_names)==2:
        input_column_name = 'input' if 'input' in column_names else column_names[0]
        target_column_name = 'target' if 'target' in column_names else column_names[0]
        train_on_inputs=False
    else:
        raise ValueError('输入文件列数不对')
    print('train_on_inputs',train_on_inputs)
    # 翻译：由于这将被pickle以避免哈希器中的_LazyModule错误，因此在tokenize_function之前强制加载记录器
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples): # 标记化函数
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer([ item for item in examples[text_column_name]],truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
            output['labels'] = output['input_ids'].copy()
        return output

    def tokenize(prompt): # 标记化函数
        result = tokenizer(prompt,truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point): # 生成并标记化提示
        input_text = data_point[input_column_name]
        target_text = data_point[target_column_name]
        full_prompt = input_text+target_text
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = input_text
            tokenized_user_prompt = tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ] 
        return tokenized_full_prompt
    
    
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function if train_on_inputs==True else generate_and_tokenize_prompt,
                batched=True if train_on_inputs==True else False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function if train_on_inputs==True else generate_and_tokenize_prompt,
                batched=True if train_on_inputs==True else False,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # 翻译：根据模型和配置的不同，logits可能包含额外的张量，如past_key_values，但logits始终在第一位
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy.py")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # 翻译：preds的shape与标签相同，在preprocess_logits_for_metrics计算argmax(-1)之后，但我们需要shift标签
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

    # 初始化训练器
    trainer = Trainer(
        model=model, # 模型
        args=training_args, # 训练参数
        train_dataset=train_dataset if training_args.do_train else None, # 训练数据集
        eval_dataset=eval_dataset if training_args.do_eval else None, # 评估数据集
        tokenizer=tokenizer, # tokenizer
        # 数据收集器将默认为DataCollatorWithPadding，因此我们将其更改为DataCollatorForSeq2Seq
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True # 数据收集器
        ),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None, # 计算指标
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available()else None, # 预处理logits
    )

    # 开始训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        print(training_args.local_rank,'start train')
        
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 开始评估
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()