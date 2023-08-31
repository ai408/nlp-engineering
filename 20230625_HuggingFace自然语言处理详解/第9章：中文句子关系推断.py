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


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\ChnSentiCorp'
        dataset = load_from_disk(pretrained_model_name_or_path)[split]
        # 过滤长度大于40的句子
        self.dataset = dataset.filter(lambda data: len(data['text']) > 40)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        text = self.dataset[i]['text']
        # 将一句话切分为前半句和后半句
        sentence1 = text[:20]
        sentence2 = text[20:40]
        # 随机整数，取值为0和1
        label = random.randint(0, 1)
        # 有一半概率把后半句替换为无关的句子
        if label == 1:
            j = random.randint(0, len(self.dataset) - 1) # 随机取出一句话
            sentence2 = self.dataset[j]['text'][20:40] # 取出后半句
        return sentence1, sentence2, label # 返回前半句、后半句和标签


# 数据整理函数
def collate_fn(data):
    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents, # 输入句子对
                                   truncation=True, # 截断
                                   padding='max_length', # [PAD]
                                   max_length=45, # 最大长度
                                   return_tensors='pt', # 返回pytorch张量
                                   return_length=True, # 返回长度
                                   add_special_tokens=True) # 添加特殊符号
    # input_ids：编码之后的数字
    # attention_mask：补零的位置是0, 其他位置是1
    # token_type_ids：第1个句子和特殊符号的位置是0, 第2个句子的位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型抽取数据特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 对抽取的特征只取第1个字的结果进行分类即可
        out = self.fc(out.last_hidden_state[:, 0, :])
        out = out.softmax(dim=1)
        return out


# 训练
def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # 定义1oss函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(loader), optimizer=optimizer)
    # 将模型切换到训练模式
    model.train()
    # 按批次遍历训练集中的数据
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
            if i % 20 == 0:
                out = out.argmax(dim=1) # 取出最大值的索引
                accuracy = (out == labels).sum().item() / len(labels) # 计算准确率
                lr = optimizer.state_dict()['param_groups'][0]['lr'] # 获取当前学习率
                print(epoch, 1, loss.item(), lr, accuracy)


# 测试
def test():
    # 定义测试数据集加载器
    dataset = Dataset('test')
    loader_test = torch.utils.data.DataLoader(dataset=dataset,  batch_size=32, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # 将下游任务模型切换到运行模式
    model.eval()
    correct = 0
    total = 0
    # 按批次遍历测试集中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        # 计算5个批次即可，不需要全部遍历
        if i == 5:
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
    dataset = Dataset('train')
    # sentence1, sentence2, label = dataset[7]
    # print(len(dataset), sentence1, sentence2, label)
    # print(sentence2)


    # ###################################################################################################################
    # 定义计算设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # print(device)



    ###################################################################################################################
    # 测试数据整理函数
    # data = [('酒店还是非常的不错，我预定的是套间，服务', '非常好，随叫随到，结账非常快。',
    #          0),
    #         ('外观很漂亮，性价比感觉还不错，功能简', '单，适合出差携带。蓝牙摄像头都有了。',
    #          0),
    #         ('《穆斯林的葬礼》我已闻名很久，只是一直没', '怎能享受4星的服务，连空调都不能用的。', 1)]
    # input_ids, attention_mask, token_type_ids, labels = collate_fn(data)
    # # 把编码还原为句子
    # print(token.decode(input_ids[0]))
    # print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

    # 数据集加载器
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn, shuffle=True, drop_last=True)
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