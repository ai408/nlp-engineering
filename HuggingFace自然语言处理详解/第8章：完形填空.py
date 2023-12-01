"""
1.文本数据集的下载与各种操作：https://blog.csdn.net/Wang_Dou_Dou_/article/details/127459760
"""
from pathlib import Path
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
from transformers.optimization import get_scheduler, AdamW
import torch
from datasets import load_from_disk
import random


def load_encode_tool(pretrained_model_name_or_path):
    token = BertTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    return token


# 编码
def f1(data):
    # 通过编码工具将文字编码为数据，同时删除多余字段
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\bert-base-chinese'
    token = AutoTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'))

    result = token.batch_encode_plus(batch_text_or_text_pairs=data['text'],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=30,
                                   return_length=True)

    # 打印解码结果
    # print(token.decode(result['input_ids'][0]))

    return result


# def f2(data):
#     # 过滤掉太短的句子
#     return [i >= 30 for i in data['length']]


def load_dataset_from_disk():
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\ChnSentiCorp'
    dataset = load_from_disk(pretrained_model_name_or_path)
    # batched=True表示批量处理
    # batch_size=1000表示每次处理1000个样本
    # num_proc=8表示使用8个线程操作
    # remove_columns=['text']表示移除text列
    dataset = dataset.map(f1, batched=True, batch_size=1000, num_proc=8, remove_columns=['text', 'label'])

    return dataset


# 数据整理函数
def collate_fn(data):
    # 取出编码结果
    input_ids = [i['input_ids'] for i in data]
    attention_mask = [i['attention_mask'] for i in data]
    token_type_ids = [i['token_type_ids'] for i in data]
    # 转换为Tensor格式
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    # 把第15个词替换为MASK
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = token.get_vocab()[token.mask_token]
    # 移动到计算设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(in_features=768,
                                       out_features=token.vocab_size,
                                       bias=False)
        # 重新将decode中的bias参数初始化为全o
        self.bias = torch.nn.Parameter(data=torch.zeros(token.vocab_size))
        self.decoder.bias = self.bias
        # 定义Dropout层，防止过拟合
        self.Dropout = torch.nn.Dropout(p=0.5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型抽取数据特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 把第15个词的特征投影到全字典范围内
        out = self.Dropout(out.last_hidden_state[:, 15])
        out = self.decoder(out)
        return out


# 训练
def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1.0)
    # 定义1oss函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(loader) * 5, optimizer=optimizer)
    # 将模型切换到训练模式
    model.train()
    # 共训练5个epoch
    for epoch in range(5):
        # 按批次遍历训练集中的数据
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            # 模型计算
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # 计算loss并使用梯度下降法优化模型参数
            loss = criterion(out, labels)
            loss.backward() # 反向传播
            optimizer.step() # 梯度下降法优化模型参数
            scheduler.step() # 学习率调节器
            optimizer.zero_grad() # 清空梯度
            # 输出各项数据的情况，便于观察
            if i % 50 == 0:
                out = out.argmax(dim=1) # 取出最大值的索引
                accuracy = (out == labels).sum().item() / len(labels) # 计算准确率
                lr = optimizer.state_dict()['param_groups'][0]['lr'] # 获取当前学习率
                print(epoch, 1, loss.item(), lr, accuracy)


# 测试
def test():
    # 定义测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=dataset['test'],  batch_size=32, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # 将下游任务模型切换到运行模式
    model.eval()
    correct = 0
    total = 0
    # 按批次遍历测试集中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        # 计算15个批次即可，不需要全部遍历
        if i == 15:
            break
        print(i)
        # 计算
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 统计正确率
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
    print(correct / total)


if __name__ == '__main__':
    ###################################################################################################################
    # 测试编码工具
    pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\bert-base-chinese'
    token = load_encode_tool(pretrained_model_name_or_path)
    # print(token)
    # 测试编码句子
    # out = token.batch_encode_plus(
    #     batch_text_or_text_pairs=[('不是一切大树，', '都被风暴折断。'),('不是一切种子，', '都找不到生根的土壤。')],
    #     truncation=True,
    #     padding='max_length',
    #     max_length=18,
    #     return_tensors='pt',
    #     return_length=True, # 返回长度
    # )
    # # 查看编码输出
    # for k, v in out.items():
    #     print(k, v.shape)
    # print(token.decode(out['input_ids'][0]))
    # print(token.decode(out['input_ids'][1]))


    # ###################################################################################################################
    # 加载数据集
    dataset = load_dataset_from_disk()
    # print(dataset) # 9600个训练样本，1200个验证样本，1200个测试样本
    # dataset = dataset.filter(f2, batched=True, batch_size=1000, num_proc=8)
    # print(dataset) # 9600个训练样本，1200个验证样本，1200个测试样本


    # ###################################################################################################################
    # 定义计算设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # print(device)


    # ###################################################################################################################
    # 数据集加载器
    loader = torch.utils.data.DataLoader(dataset=dataset['train'], batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # print(len(loader))

    # 查看数据样例
    # for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    #     break
    # print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)


    # ###################################################################################################################
    # 加载预训练模型
    pretrained = BertModel.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    # 统计参数量
    # print(sum(i.numel() for i in pretrained.parameters()) / 10000)

    # 不训练预训练模型，不需要计算梯度
    # for param in pretrained.parameters():
    #     param.requires_grad_(False)

    # 测试预训练模型
    pretrained.to(device)
    # out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # print(out.last_hidden_state.shape)

    model = Model()
    model.to(device)
    # print(model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).shape)


    # ###################################################################################################################
    # 训练和测试模型
    train()
    test()