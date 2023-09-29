import torch
import pandas as pd
import numpy as np
import random


# 定义数据集
class MsrDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data.iloc[i]


# 定义数据整理函数
def collate_fn(data):
    # 取出数据
    same = [i['same'] for i in data]
    sent = [i['sent'] for i in data]
    s1_lens = [i['s1_lens'] for i in data]
    s2_lens = [i['s2_lens'] for i in data]
    pad_lens = [i['pad_lens'] for i in data]
    seg = []
    for i in range(len(sent)):
        # seg的形状和sent一样，但是内容不一样
        # 补PAD的位置是0，s1的位置是1，s2的位置是2
        seg.append([1] * s1_lens[i] + [2] * s2_lens[i] + [0] * pad_lens[i])
    #  sent由字符型转换为list
    sent = [np.array(j.split(','), dtype=np.int32) for j in sent]
    same = torch.LongTensor(same)
    sent = torch.LongTensor(sent)
    seg = torch.LongTensor(seg)
    return same, sent, seg


# 定义随机替换函数
def random_replace(sent):
    # sent = [b,63]
    # 不影响原来的sent
    sent = sent.clone()
    # 替换矩阵，形状和sent一样，被替换过的位置是True，其他位置是False
    replace = sent == -1
    # 遍历所有的词
    for i in range(len(sent)):
        for j in range(len(sent[i])):
            # 如果是符号就不操作了，只替换词
            if sent[i, j] <= 10:
                continue
            # 以0.15的概率进行操作
            if random.random() > 0.15:
                pass
            # 对被操作过的位置进行标记，这里的操作包括什么也不做
            replace[i, j] = True
            # 分概率做不同的操作
            p = random.random()
            # 以O.8的概率替换为MASK
            if p < 0.8:
                sent[i, j] = vocab.loc['<MASK>'].token
            # 以0.1的概率不替换
            elif p < 0.9:
                continue
            # 以0.1的概率替换成随机词
            else:
                # 随机生成一个不是符号的词
                rand_word = 0
                while rand_word <= 10:
                    rand_word = random.randint(0, len(vocab) - 1)
                sent[i, j] = rand_word
    return sent, replace


# 定义获取MASK的函数
def get_mask(seg):
    # key_padding_mask的定义方式为句子中PAD的位置为True，否则为False
    key_padding_mask = seg == 0
    # 在encode阶段不需要定义encode_attn_mask
    # 定义为None或者全False都可以
    encode_attn_mask = torch.ones(63, 63) == -1
    return key_padding_mask, encode_attn_mask


# 定义模型
class BERTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义词向量编码层
        self.sent_embed = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=256)
        # 定义seg编码层
        self.seg_embed = torch.nn.Embedding(num_embeddings=3, embedding_dim=256)
        # 定义位置编码层
        self.position_embed = torch.nn.Parameter(torch.randn(63, 256) / 10)
        # 定义编码层
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=256, dropout=0.2, activation='relu', batch_first=True, norm_first=True)
        # 定义标准化层
        norm = torch.nn.LayerNorm(normalized_shape=256, elementwise_affine=True)
        # 定义编码器
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4, norm=norm)
        # 定义same输出层
        self.fc_same = torch.nn.Linear(in_features=256, out_features=2)
        # 定义sent输出层
        self.fc_sent = torch.nn.Linear(in_features=256, out_features=len(vocab))

    def forward(self, sent, seg):
        # sent -> [b,63]
        # seg -> [b,63]
        # 获取MASK
        # [b, 63]-> [b, 63], [63, 63]
        key_padding_mask, encode_attn_mask = get_mask(seg)
        # 编码，添加位置信息
        # [b, 63] -> [b, 63, 256]
        embed = self.sent_embed(sent) + self.seg_embed(seg) + self.position_embed
        # 编码器计算
        # [b,63,256] -> [b,63,256]
        memory = self.encoder(src=embed, mask=encode_attn_mask, src_key_padding_mask=key_padding_mask)
        # 计算输出，same的输出使用第0个词的信息计算
        # [b, 256] -> [b,2]
        same = self.fc_same(memory[:, 0])
        # [b, 63, 256] -> [b, 63, V]
        sent = self.fc_sent(memory)
        return same, sent


# 训练
def train():
    loss_func = torch.nn.CrossEntropyLoss() # 交叉熵损失函数
    optim = torch.optim.Adam(model.parameters(), lr=1e-4) # Adam优化器
    for epoch in range(2000):
        for i, (same, sent, seg) in enumerate(loader_train):
            # same = [b]
            # sent = [b,63]
            # seg = [b, 63]
            # 随机替换x中的某些字符，replace为是否被操作过的矩阵，这里的操作包括不替换
            # replace_sent = [b, 63]
            # replace = [b,63]
            replace_sent, replace = random_replace(sent) # replace_sent是替换后的句子，replace是替换的位置
            # 模型计算
            # [b,63],[b,63] -> (b,2],[b,63,V]
            pred_same, pred_sent = model(replace_sent, seg)
            # 只把被操作过的字提取出来
            # [b, 63, V] -> [replace, V]
            pred_sent = pred_sent[replace]
            # 把被操作之前的字提取出来
            # [b, 63] -> [replace]
            sent = sent[replace]
            # 计算两份loss，再加权求和
            loss_same = loss_func(pred_same, same)
            loss_sent = loss_func(pred_sent, sent)
            loss = loss_same * 0.01 + loss_sent
            loss.backward() # 反向传播
            optim.step() # 更新参数
            optim.zero_grad() # 清空梯度
        if epoch % 5 == 0:
            # 计算same预测正确率
            pred_same = pred_same.argmax(dim=1)
            acc_same = (same == pred_same).sum().item() / len(same)
            # 计算替换词预测正确率
            pred_sent = pred_sent.argmax(dim=1)
            acc_sent = (sent == pred_sent).sum().item() / len(sent)
            print(epoch, i, loss.item(), acc_same, acc_sent)


# 定义工具函数，把Tensor转换为字符串
def tensor_to_str(tensor):
    # 转换为list格式
    tensor = tensor.tolist()
    # 过滤掉PAD
    tensor = [i for i in tensor if i != vocab.loc['<PAD>'].token]
    # 转换为词
    tensor = [vocab_r.loc[i].word for i in tensor]
    # 转换为字符串
    return ' '.join(tensor)


# 定义工具函数，打印预测结果
def print_predict(same, pred_same, replace_sent, sent, pred_sent, replace):
    # 输出same预测结果
    same = same[0].item()
    pred_same = pred_same.argmax(dim=1)[0].item()
    print('same=', same, 'pred_same=', pred_same)
    print()
    # 输出句子替换词的预测结果
    replace_sent = tensor_to_str(replace_sent[0])
    sent = tensor_to_str(sent[0][replace[0]])
    pred_sent = tensor_to_str(pred_sent.argmax(dim=2)[0][replace[0]])
    print('replace_sent=', replace_sent)
    print()
    print('sent=', sent)
    print()
    print('pred_sent=', pred_sent)
    print()
    print('-------------------------------------')


# 测试
def model_test():
    model.eval() # 模型进入测试模式
    correct_same = 0 # same预测正确的数量
    total_same = 0 # same总数量
    correct_sent = 0 # sent预测正确的数量
    total_sent = 0 # sent总数量
    for i, (same, sent, seg) in enumerate(loader_test):
        # 测试5个批次
        if i == 5:
            break
        # same = [b]
        # sent=[b,63]
        # seg = [b, 63]
        # 随机替换x中的某些字符，replace为是否被操作过的矩阵，这里的操作包括不替换
        # replace_sent = [b,63]
        # replace = [b, 63]
        replace_sent, replace = random_replace(sent)
        # 模型计算
        # [b,63],[b,63]->[b,2],[b,63,V]
        with torch.no_grad(): # 不计算梯度
            pred_same, pred_sent = model(replace_sent, seg)
        # 输出预测结果
        print_predict(same, pred_same, replace_sent, sent, pred_sent, replace)
        # 只把被操作过的字提取出来
        # [b,63, V] -> [replace,V]
        pred_sent = pred_sent[replace]
        # 把被操作之前的字取出来
        # [b, 63] -> [replace]
        sent = sent[replace]
        # 计算same的预测正确率
        pred_same = pred_same.argmax(dim=1)
        correct_same += (same == pred_same).sum().item()
        total_same += len(same)
        # 计算替换词的预测正确率
        pred_sent = pred_sent.argmax(dim=1)
        correct_sent += (sent == pred_sent).sum().item()
        total_sent += len(sent)
    print(correct_same / total_same)
    print(correct_sent / total_sent)


if __name__ == '__main__':
    # 1.读取字典
    vocab = pd.read_csv('dataset/msr_paraphrase_vocab.csv', index_col='word') # word列，即词
    vocab_r = pd.read_csv('dataset/msr_paraphrase_vocab.csv', index_col='token') # token列，即词的编号
    # print(vocab, vocab_r)

    # 2.读取数据集
    dataset_train = MsrDataset('dataset/msr_paraphrase_data_train.csv')
    dataset_test = MsrDataset('dataset/msr_paraphrase_data_test.csv')
    # print(len(dataset_train), dataset_train[0])
    # print(len(dataset_test), dataset_test[0])

    # 3.定义数据整理函数
    # print(collate_fn([dataset[0], dataset[1]]))

    # 4.定义数据集加载器
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=320, shuffle=True, drop_last=True, collate_fn=collate_fn)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=320, shuffle=True, drop_last=True, collate_fn=collate_fn)
    # print(len(loader_train))
    # print(len(loader_test))

    # 5.查看数据样例
    # for i, (same, sent, seg) in enumerate(loader_train):
    #     break

    # 6.定义随机替换函数
    # replace_sent, replace = random_replace(sent)
    # print(replace_sent) # 替换后的句子
    # print(replace) # 替换的位置
    # print(replace_sent[replace]) # 被替换的词

    # 7.定义MASK函数
    # key_padding_mask, encode_attn_mask = get_mask(seg)
    # print(key_padding_mask.shape, encode_attn_mask.shape, key_padding_mask[0], encode_attn_mask)

    # 8.定义BERT模型
    model = BERTModel()
    # pred_same, pred_sent = model(sent, seg)
    # print(pred_same.shape, pred_sent.shape)

    # 9.定义工具函数：把Tensor转换为字符串
    # print(tensor_to_str(sent[0]))
    # 10.定义工具函数：打印预测结果
    # print_predict(same, torch.randn(32, 2), replace_sent, sent, torch.randn(32, 63, 100), replace)

    # 11.训练
    train()
    # 保存模型
    torch.save(model.state_dict(), 'model/bert_model.pth')

    # 12.测试
    # 加载模型
    # model.load_state_dict(torch.load('model/bert_model.pth'))
    # model_test()