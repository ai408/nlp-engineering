"""
1.文本数据集的下载与各种操作：https://blog.csdn.net/Wang_Dou_Dou_/article/details/127459760
"""
from pathlib import Path
from transformers import BertTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from datasets import load_from_disk
from transformers import AdamW
from transformers.optimization import get_scheduler


def load_encode_tool(pretrained_model_name_or_path):
    """
    加载编码工具
    """
    tokenizer = BertTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    return tokenizer


# 加载数据集
def load_dataset_from_disk():
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\ChnSentiCorp'
    dataset = load_from_disk(pretrained_model_name_or_path)
    return dataset


# 数据整理函数
def collate_fn(data):
    sents = [i['text'] for i in data]
    labels = [i['label'] for i in data]
    #编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents, # 输入文本
            truncation=True, # 是否截断
            padding=True, # 是否填充
            max_length=512, # 最大长度
            return_tensors='pt') # 返回的类型
    #转移到计算设备
    for k, v in data.items():
        data[k] = v.to(device)
    data['labels'] = torch.LongTensor(labels).to(device)
    return data


def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-4)
    # 定义学习率调节器
    scheduler = get_scheduler(name='linear', # 调节器名称
                              num_warmup_steps=0, # 预热步数
                              num_training_steps=len(loader), # 训练步数
                              optimizer=optimizer) # 优化器
    # 将模型切换到训练模式
    model.train()
    # 按批次遍历训练集中的数据
    for i, data in enumerate(loader):
        # print(i, data)
        # 模型计算
        out = model(**data)
        # 计算1oss并使用梯度下降法优化模型参数
        out['loss'].backward() # 反向传播
        optimizer.step() # 优化器更新
        scheduler.step() # 学习率调节器更新
        optimizer.zero_grad() # 梯度清零
        model.zero_grad() # 梯度清零
        # 输出各项数据的情况，便于观察
        if i % 10 == 0:
            out_result = out['logits'].argmax(dim=1)
            accuracy = (out_result == data.labels).sum().item() / len(data.labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(i, out['loss'].item(), lr, accuracy)


def test():
    # 定义测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=dataset['test'],
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)
    # 将下游任务模型切换到运行模式
    model.eval()
    correct = 0
    total = 0
    # 按批次遍历测试集中的数据
    for i, data in enumerate(loader_test):
        # 计算5个批次即可，不需要全部遍历
        if i == 5:
            break
        print(i)
        # 计算
        with torch.no_grad():
            out = model(**data)
        # 统计正确率
        out = out['logits'].argmax(dim=1)
        correct += (out == data.labels).sum().item()
        total += len(data.labels)
    print(correct / total)


if __name__ == '__main__':
    ###################################################################################################################
    # 测试编码工具
    pretrained_model_name_or_path = r'L:/20230713_HuggingFaceModel/bert-base-chinese'
    tokenizer = load_encode_tool(pretrained_model_name_or_path)
    # print(tokenizer)


    # ###################################################################################################################
    # 加载数据集
    dataset = load_dataset_from_disk()
    # print(dataset)


    # ###################################################################################################################
    # 定义计算设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # print(device)


    ###################################################################################################################
    # 数据集加载器
    loader = torch.utils.data.DataLoader(dataset=dataset['train'], batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # print(len(loader))

    # 查看数据样例
    # for i, data in enumerate(loader):
    #     break
    # for k, v in data.items():
    #     print(k, v.shape)


    # ###################################################################################################################
    # 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(Path(f'{pretrained_model_name_or_path}'), num_labels=2)
    model.to(device)
    # print(sum(i.numel() for i in model.parameters()) / 10000)

    # 模型测试
    # out = model(**data)
    # print(out['loss'], out.logits.shape)


    ###################################################################################################################
    # 训练和测试模型
    train()
    test()