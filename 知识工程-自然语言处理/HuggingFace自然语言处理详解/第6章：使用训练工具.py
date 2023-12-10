"""
1.文本数据集的下载与各种操作：https://blog.csdn.net/Wang_Dou_Dou_/article/details/127459760
"""
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import TrainingArguments
import torch

def load_encode():
    # 1.加载编码工具
    # 加载tokenizer
    from transformers import AutoTokenizer
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\rbt3'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    # 试编码句子
    result = tokenizer.batch_encode_plus(
        ['明月装饰了你的窗子', '你装饰了别人的梦'],
        truncation=True,
    )
    print(result)
    return tokenizer


#编码
def f1(data):
    # 通过编码工具将文字编码为数据
    from transformers import AutoTokenizer
    from pathlib import Path
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\rbt3'
    tokenizer = AutoTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    return tokenizer.batch_encode_plus(data['text'], truncation=True)

def f2(data):
    # 过滤太长的句子
    return [len(i) <= 512 for i in data['input_ids']]

def load_dataset_from_disk():
    # 方法1：从HuggingFace加载数据集，然后本地保存
    # from datasets import load_dataset
    # dataset = load_dataset(path='seamew/ChnSentiCorp')
    # print(dataset)
    # dataset.save_to_disk(dataset_dict_path='./data/ChnSentiCorp')

    # 方法2：从本地加载数据集
    from datasets import load_from_disk
    mode_name_or_path = r'L:\20230713_HuggingFaceModel\ChnSentiCorp'
    dataset = load_from_disk(mode_name_or_path)
    # 缩小数据规模，便于测试
    dataset['train'] = dataset['train'].shuffle().select(range(2000))
    dataset['test'] = dataset['test'].shuffle().select(range(100))

    # batched=True表示批量处理
    # batch_size=1000表示每次处理1000个样本
    # num_proc=4表示使用4个线程操作
    # remove_columns=['text']表示移除text列
    dataset = dataset.map(f1, batched=True, batch_size=1000, num_proc=4, remove_columns=['text'])

    return dataset


def load_pretrained_mode():
    """
    加载预训练模型
    """
    from transformers import AutoModelForSequenceClassification
    import torch
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\rbt3'
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=2)
    # 统计模型参数量
    print(sum([i.nelement() for i in model.parameters()]) / 10000)

    # 模拟一批数据
    data = {
        'input_ids': torch.ones(4, 10, dtype=torch.long),
        'token_type_ids': torch.ones(4, 10, dtype=torch.long),
        'attention_mask': torch.ones(4, 10, dtype=torch.long),
        'labels': torch.ones(4, dtype=torch.long)
    }
    # 模型试算
    out = model(**data)
    print(out['loss'], out['logits'].shape)

    return model


def compute_metrics(eval_pred):
    """
    定义评价函数
    """
    from datasets import load_metric
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    logits = logits.argmax(axis=1)
    return metric.compute(predictions=logits, references=labels)


def DataCollator_test(tokenizer, dataset):
    """
    数据整理函数
    """
    from transformers import DataCollatorWithPadding
    # 测试数据整理函数
    data_collator = DataCollatorWithPadding(tokenizer)
    # 获取一批数据
    data = dataset['train'][:10]
    # 输出这些句子的长度
    for i in data['input_ids']:
        print(len(i))
    # 调用数据整理函数
    data = data_collator(data)
    # 查看整理后的数据
    for k, v in data.items():
        print(k, v.shape)


if __name__ == '__main__':
    ###################################################################################################################
    # 主要任务：训练模型
    # 加载编码工具
    tokenizer = load_encode()

    # 没有移除太长句子
    dataset = load_dataset_from_disk()
    # 过滤太长的句子
    dataset = dataset.filter(f2, batched=True, batch_size=1000, num_proc=4)
    # print(dataset)

    # 加载预训练模型
    model = load_pretrained_mode()

    # 定义训练超参数
    args = TrainingArguments(
        # 定义临时数据保存路径
        output_dir='./output_dir',
        # 定义测试执行的策略，可取值为no、epoch、steps
        evaluation_strategy='steps',
        # 定义每隔多少个step执行一次测试
        eval_steps=30,
        # 定义模型保存策略，可取值为no、epoch、steps
        save_strategy='steps',
        # 定义每隔多少个step保存一次
        save_steps=30,
        # 定义共训练几个轮次
        num_train_epochs=1,
        # 定义学习率
        learning_rate=1e-4,
        # 加入参数权重衰减，防止过拟合
        weight_decay=1e-2,
        # 定义测试和训练时的批次大小
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        # 定义是否要使用GPU训练
        no_cuda=False,
    )

    # 定义训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # print(trainer.evaluate())
    # trainer.train()

    ###################################################################################################################
    # 从某个存档文件继续训练
    # trainer.train(resume_from_checkpoint='./output_dir/checkpoint-90')

    # 手动保存模型参数
    # trainer.save_model(output_dir='./output_dir/save_model')
    # 手动加载模型参数
    model.load_state_dict(torch.load('./output_dir/save_model/PyTorch_model.bin'))

    # 模型测试
    # 在模型的评估模式下，模型不再对输入进行梯度计算，并且一些具有随机性的操作（如Dropout）会被固定
    model.eval()
    for i, data in enumerate(trainer.get_eval_dataloader()):
        data = data.to('cuda')
        out = model(**data)
        out = out['logits'].argmax(dim=1)
        for j in range(16):
            print(tokenizer.decode(data['input_ids'][j], skip_special_tokens=True))
            print('label=', data['labels'][j].item())
            print('predict=', out[j].item())
        break


    ####################################################################################################################
    # 测试定义评价函数
    # from transformers.trainer_utils import EvalPrediction
    # import numpy as np
    # eval_pred = EvalPrediction(
    #     predictions=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
    #     label_ids=np.array([1, 1, 0, 1]),
    # )
    # accuracy = compute_metrics(eval_pred)
    # print(accuracy)


    ####################################################################################################################
    # 主要任务：测试数据整理函数：得到tokenizer和dataset
    # from transformers import AutoTokenizer
    # from pathlib import Path
    # pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\rbt3'
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=Path(f'{pretrained_model_name_or_path}'))
    # dataset = load_dataset_from_disk()
    # dataset = dataset.filter(f2, batched=True, batch_size=1000, num_proc=4)
    # DataCollator_test(tokenizer, dataset)