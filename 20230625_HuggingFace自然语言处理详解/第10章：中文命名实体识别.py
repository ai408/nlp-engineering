"""
1.文本数据集的下载：
"""
from pathlib import Path
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel, AutoModel
import torch
from datasets import load_dataset, load_from_disk
from transformers import AdamW
from transformers.optimization import get_scheduler


def load_encode_tool(pretrained_model_name_or_path):
    """
    加载编码工具
    """
    tokenizer = AutoTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    return tokenizer


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        # 在线加载数据集
        # dataset = load_dataset(path='people_daily_ner', split=split)
        # dataset.save_to_disk(dataset_dict_path='L:/20230713_HuggingFaceModel/peoples_daily_ner')
        # 离线加载数据集
        dataset = load_from_disk(dataset_path='L:/20230713_HuggingFaceModel/peoples_daily_ner')[split]
        # print(dataset.features['ner_tags'].feature.num_classes) #7
        # print(dataset.features['ner_tags'].feature.names) # ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]['tokens']
        labels = self.dataset[i]['ner_tags']
        return tokens, labels



# 数据整理函数
def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]
    inputs = tokenizer.batch_encode_plus(tokens, # 文本列表
                                   truncation=True, # 截断
                                   padding=True, # [PAD]
                                   max_length=512, # 最大长度
                                   return_tensors='pt', # 返回pytorch张量
                                   is_split_into_words=True) # 按词切分
    # 求一批数据中最长的句子长度
    lens = inputs['input_ids'].shape[1]
    # 在labels的头尾补充7，把所有的labels补充成统一的长度
    for i in range(len(labels)):
        labels[i] = [7] + labels[i]
        labels[i] += [7] * lens
        labels[i] = labels[i][:lens]
    # 把编码结果移动到计算设备上
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    # 把统一长度的labels组装成矩阵，移动到计算设备上
    labels = torch.tensor(labels).to(device)
    return inputs, labels


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 标识当前模型是否处于tuning模式
        self.tuning = False
        # 当处于tuning模式时backbone应该属于当前模型的一部分，否则该变量为空
        self.pretrained = None
        # 当前模型的神经网络层
        self.rnn = torch.nn.GRU(input_size=768, hidden_size=768, batch_first=True)
        self.fc = torch.nn.Linear(in_features=768, out_features=8)

    def forward(self, inputs):
        # 根据当前模型是否处于tuning模式而使用外部backbone或内部backbone计算
        if self.tuning:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state
        # backbone抽取的特征输入RNN网络进一步抽取特征
        out, _ = self.rnn(out)
        # RNN网络抽取的特征最后输入FC神经网络分类
        out = self.fc(out).softmax(dim=2)
        return out

    # 切换下游任务模型的tuning模式
    def fine_tuning(self, tuning):
        self.tuning = tuning
        # tuning模式时，训练backbone的参数
        if tuning:
            for i in pretrained.parameters():
                i.requires_grad = True
            pretrained.train()
            self.pretrained = pretrained
        # 非tuning模式时，不训练backbone的参数
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)
            pretrained.eval()
            self.pretrained = None


# 对计算结果和labels变形，并且移除PAD
def reshape_and_remove_pad(outs, labels, attention_mask):
    # 变形，便于计算loss
    # [b, lens, 8] -> [b*lens, 8]
    outs = outs.reshape(-1, 8)
    # [b, lens] -> [b*lens]
    labels = labels.reshape(-1)
    # 忽略对PAD的计算结果
    # [b, lens] -> [b*lens - pad]
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]
    return outs, labels


# 获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    # [b*lens, 8] -> [b*lens]
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)
    # 计算除了0以外元素的正确率，因为0太多了，所以正确率很容易虚高
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)
    return correct, total, correct_content, total_content


# 训练
def train(epochs):
    lr = 2e-5 if model.tuning else 5e-4 # 根据模型的tuning模式设置学习率
    optimizer = AdamW(model.parameters(), lr=lr) # 优化器
    criterion = torch.nn.CrossEntropyLoss() # 损失函数
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(loader) * epochs, optimizer=optimizer) # 学习率衰减策略
    model.train()
    for epoch in range(epochs):
        for step, (inputs, labels) in enumerate(loader):
            # 模型计算
            # [b,lens] -> [b,lens,8]
            outs = model(inputs)
            # 对outs和labels变形，并且移除PAD
            # outs -> [b, lens, 8] -> [c, 8]
            # labels -> [b, lens] -> [c]
            outs, labels = reshape_and_remove_pad(outs, labels, inputs['attention_mask'])
            # 梯度下降
            loss = criterion(outs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            scheduler.step() # 更新学习率
            optimizer.zero_grad() # 清空梯度
            if step % (len(loader) * epochs // 30) == 0:
                counts = get_correct_and_total_count(labels, outs)
                accuracy = counts[0] / counts[1]
                accuracy_content = counts[2] / counts[3]
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(epoch, step, loss.item(), lr, accuracy, accuracy_content)
    torch.save(model, 'model/中文命名实体识别.model')



# 测试
def test():
    # 加载训练完的模型
    model_load = torch.load('model/中文命名实体识别.model')
    model_load.eval() # 切换到评估模式
    model_load.to(device)
    # 测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'), batch_size=128, collate_fn=collate_fn, shuffle=True, drop_last=True)
    correct = 0
    total = 0
    correct_content = 0
    total_content = 0
    # 遍历测试数据集
    for step, (inputs, labels) in enumerate(loader_test):
        # 测试5个批次即可，不用全部遍历
        if step == 5:
            break
        print(step)
        # 计算
        with torch.no_grad():
            # [b, lens] -> [b, lens, 8] -> [b, lens]
            outs = model_load(inputs)
        # 对outs和labels变形，并且移除PAD
        # fouts -> [b, lens, 8] -> [c, 8]
        # labels -> [b, lens] -> [c]
        outs, labels = reshape_and_remove_pad(outs, labels, inputs['attention_mask'])
        # 统计正确数量
        counts = get_correct_and_total_count(labels, outs)
        correct += counts[0]
        total += counts[1]
        correct_content += counts[2]
        total_content += counts[3]
    print(correct / total, correct_content / total_content)



# 预测
def predict():
    # 加载模型
    model_load = torch.load('model/中文命名实体识别.model')
    model_load.eval()
    model_load.to(device)
    # 测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'), batch_size=32, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # 取一个批次的数据
    for i, (inputs, labels) in enumerate(loader_test):
        break
    # 计算
    with torch.no_grad():
        # [b, lens] -> [b, lens, 8] -> [b, lens]
        outs = model_load(inputs).argmax(dim=2)
    for i in range(32):
        # 移除PAD
        select = inputs['attention_mask'][i] == 1
        input_id = inputs['input_ids'][i, select]
        out = outs[i, select]
        label = labels[i, select]
        # 输出原句子
        print(tokenizer.decode(input_id).replace(' ', ''))
        # 输出tag
        for tag in [label, out]:
            s = ''
            for j in range(len(tag)):
                if tag[j] == 0:
                    s += '.'
                    continue
                s += tokenizer.decode(input_id[j])
                s += str(tag[j].item())
            print(s)
        print('=====================')


if __name__ == '__main__':
    ###################################################################################################################
    # 测试编码工具
    pretrained_model_name_or_path = r'L:/20230713_HuggingFaceModel/rbt3'
    tokenizer = load_encode_tool(pretrained_model_name_or_path)
    # print(tokenizer)
    # 测试编码句子
    # out = tokenizer.batch_encode_plus(
    #     batch_text_or_text_pairs=[
    #         [ '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。'],
    #         [ '这', '座', '依', '山', '傍', '水', '的', '博', '物', '馆', '由', '国', '内', '一', '流', '的', '设', '计', '师', '主', '持', '设', '计', '。']],
    #     truncation=True, # 截断
    #     padding='max_length', # [PAD]
    #     max_length=20, # 最大长度
    #     return_tensors='pt', # 返回pytorch张量
    #     is_split_into_words=True # 按词切分
    # )
    # # 查看编码输出
    # for k, v in out.items():
    #     print(k, v.shape)
    # # 将编码还原为句子
    # print(tokenizer.decode(out['input_ids'][0]))
    # print(tokenizer.decode(out['input_ids'][1]))


    # ###################################################################################################################
    # 加载数据集
    dataset = Dataset('train')
    # tokens, labels = dataset[0]
    # print(tokens, labels, dataset)
    # print(len(dataset))


    # # ###################################################################################################################
    # 定义计算设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # print(device)


    # ###################################################################################################################
    # 测试数据整理函数
    # data = [
    #     (
    #         ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。'], [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]
    #     ),
    #     (
    #         ['这', '座', '依', '山', '傍', '水', '的', '博', '物', '馆', '由', '国', '内', '一', '流', '的', '设', '计', '师', '主', '持', '设', '计', ',', '整', '个', '建', '筑', '群', '精', '美', '而', '恢', '宏', '。'],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     )]
    # inputs, labels = collate_fn(data)
    # for k, v in inputs.items():
    #     print(k, v.shape)
    # print('labels', labels.shape)
    # print('labels', labels)


    # # 数据集加载器
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # print(len(loader))
    #
    # # 查看数据样例
    # for i, (inputs, labels) in enumerate(loader):
    #     break
    # print(tokenizer.decode(inputs['input_ids'][0]))
    # print(labels[0])
    # for k, v in inputs.items():
    #     print(k, v.shape)


    # ###################################################################################################################
    # 加载预训练模型
    pretrained = AutoModel.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    # 统计参数量
    # print(sum(i.numel() for i in pretrained.parameters()) / 10000)
    # 测试预训练模型
    pretrained.to(device)

    model = Model()
    model.to(device)
    # print(model(inputs).shape)


    # # ###################################################################################################################
    # 训练和测试模型
    # 测试工具函数
    # reshape_and_remove_pad(torch.randn(2, 3, 8), torch.ones(2, 3), torch.ones(2, 3))
    # get_correct_and_total_count(torch.ones(16), torch.randn(16, 8))

    # 两段式训练第一阶段，训练下游任务模型
    model.fine_tuning(False)
    # print(sum(p.numel() for p in model.parameters() / 10000))
    train(1)

    # 两段式训练第二阶段，联合训练下游任务模型和预训练模型
    model.fine_tuning(True)
    # print(sum(p.numel() for p in model.parameters() / 10000))
    train(5)

    test()
    predict()