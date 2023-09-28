import random
import numpy as np
import torch
import math


# 定义字典
vocab_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
vocab_x = {word: i for i, word in enumerate(vocab_x.split(','))}
vocab_xr = [k for k, v in vocab_x.items()]
vocab_y = {k.upper(): v for k, v in vocab_x.items()}
vocab_yr = [k for k, v in vocab_y.items()]
print('vocab_x=', vocab_x)
print('vocab_y=', vocab_y)

# 定义生成数据的函数
def get_data():
    # 定义词集合
    words = ['0','1','2','3','4','5','6','7','8','9','q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']

    # 定义每个词被选中的概率
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()

    # 随机选n个词
    n = random.randint(30, 48) # 生成30-48个词
    x = np.random.choice(words, size=n, replace=True, p=p) # words中选n个词，每个词被选中的概率为p，replace=True表示可以重复选择

    # 采样的结果就是x
    x = x.tolist()

    # y是由对x的变换得到的
    # 字母大写，数字取9以内的互补数
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)
    y = [f(i) for i in x]
    # 逆序
    y = y[::-1]
    # y中的首字母双写
    y = [y[0]] + y
    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']
    # 补PAD，直到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]
    # 编码成数据
    x = [vocab_x[i] for i in x]
    y = [vocab_y[i] for i in y]
    # 转Tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


# 定义数据集和加载器
# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self): # 初始化
        super(Dataset, self).__init__()
    def __len__(self): # 返回数据集的长度
        return 1000
    def __getitem__(self, i): # 根据索引返回数据
        return get_data()



# 定义mask_pad函数
def mask_pad(data):
    # b句话，每句话50个词，这里是还没embed的
    # data = [b, 50]
    # 判断每个词是不是<PAD>
    mask = data == vocab_x['<PAD>']
    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)
    # 在计算注意力时，计算50个词和50个词相互之间的注意力，所以是个50*50的矩阵
    # PAD的列为True，意味着任何词对PAD的注意力都是0，但是PAD本身对其它词的注意力并不是0，所以是PAD的行不为True
    # 复制n次
    # [b, 1, 1, 50] -> [b, 1, 50, 50]
    mask = mask.expand(-1, 1, 50, 50) # 根据指定的维度扩展
    return mask



# 定义mask_tril函数
def mask_tril(data):
    # b句话，每句话50个词，这里是还没embed的
    # data = [b, 50]
    # 50*50的矩阵表示每个词对其它词是否可见
    # 上三角矩阵，不包括对角线，意味着对每个词而言它只能看到它自己和它之前的词，而看不到之后的词
    # [1, 50, 50]
    """
    [[0, 1, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]]
    """
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long)) # torch.tril返回下三角矩阵，则1-tril返回上三角矩阵
    # 判断y当中每个词是不是PAD, 如果是PAD, 则不可见
    # [b, 50]
    mask = data == vocab_y['<PAD>'] # mask的shape为[b, 50]
    # 变形+转型，为了之后的计算
    # [b, 1, 50]
    mask = mask.unsqueeze(1).long() # 在指定位置插入维度，mask的shape为[b, 1, 50]
    # mask和tril求并集
    # [b, 1, 50] + [1, 50, 50] -> [b, 50, 50]
    mask = mask + tril
    # 转布尔型
    mask = mask > 0 # mask的shape为[b, 50, 50]
    # 转布尔型，增加一个维度，便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1) # mask的shape为[b, 1, 50, 50]
    return mask



# 定义注意力计算函数
def attention(Q, K, V, mask):
    """
    Q：torch.randn(8, 4, 50, 8)
    K：torch.randn(8, 4, 50, 8)
    V：torch.randn(8, 4, 50, 8)
    mask：torch.zeros(8, 1, 50, 50)
    """
    # b句话，每句话50个词，每个词编码成32维向量，4个头，每个头分到8维向量
    # Q、K、V = [b, 4, 50, 8]
    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q、K矩阵相乘，求每个词相对其它所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) # K.permute(0, 1, 3, 2)表示将K的第3维和第4维交换
    # 除以每个头维数的平方根，做数值缩放
    score /= 8**0.5
    # mask遮盖，mask是True的地方都被替换成-inf，这样在计算softmax时-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf')) # masked_fill_()函数的作用是将mask中为1的位置用value填充
    score = torch.softmax(score, dim=-1) # 在最后一个维度上做softmax
    # 以注意力分数乘以V得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)
    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)
    return score


# 多头注意力计算层
class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)   # 线性运算，维度不变
        self.fc_K = torch.nn.Linear(32, 32)   # 线性运算，维度不变
        self.fc_V = torch.nn.Linear(32, 32)   # 线性运算，维度不变
        self.out_fc = torch.nn.Linear(32, 32) # 线性运算，维度不变
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True) # 标准化
        self.DropOut = torch.nn.Dropout(p=0.1) # Dropout，丢弃概率为0.1

    def forward(self, Q, K, V, mask):
        # b句话，每句话50个词，每个词编码成32维向量
        # Q、K、V=[b,50,32]
        b = Q.shape[0] # 取出batch_size
        # 保留下原始的Q，后面要做短接（残差思想）用
        clone_Q = Q.clone()
        # 标准化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        # 线性运算，维度不变
        # [b,50,32] -> [b,50,32]
        K = self.fc_K(K) # 权重就是WK
        V = self.fc_V(V) # 权重就是WV
        Q = self.fc_Q(Q) # 权重就是WQ
        # 拆分成多个头
        # b句话，每句话50个词，每个词编码成32维向量，4个头，每个头分到8维向量
        # [b,50,32] -> [b,4,50,8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        # 计算注意力
        # [b,4,50,8]-> [b,50,32]
        score = attention(Q, K, V, mask)
        # 计算输出，维度不变
        # [b,50,32]->[b,50,32]
        score = self.DropOut(self.out_fc(score)) # Dropout，丢弃概率为0.1
        # 短接（残差思想）
        score = clone_Q + score
        return score

# 定义位置编码层
class PositionEmbedding(torch.nn.Module) :
    def __init__(self):
        super().__init__()
        # pos是第几个词，i是第几个词向量维度，d_model是编码维度总数
        def get_pe(pos, i, d_model):
            d = 1e4**(i / d_model)
            pe = pos / d
            if i % 2 == 0:
                return math.sin(pe) # 偶数维度用sin
            return math.cos(pe) # 奇数维度用cos
        # 初始化位置编码矩阵
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe. unsqueeze(0) # 增加一个维度，shape变为[1,50,32]
        # 定义为不更新的常量
        self.register_buffer('pe', pe)
        # 词编码层
        self.embed = torch.nn.Embedding(39, 32) # 39个词，每个词编码成32维向量
        # 用正太分布初始化参数
        self.embed.weight.data.normal_(0, 0.1)
    def forward(self, x):
        # [8,50]->[8,50,32]
        embed = self.embed(x)
        # 词编码和位置编码相加
        # [8,50,32]+[1,50,32]->[8,50,32]
        embed = embed + self.pe
        return embed


# 定义全连接输出层
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential( # 线性全连接运算
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),)
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
    def forward(self, x):
        # 保留下原始的x，后面要做短接（残差思想）用
        clone_x = x.clone()
        # 标准化
        x = self.norm(x)
        # 线性全连接运算
        # [b,50,32]->[b,50,32]
        out = self.fc(x)
        # 做短接（残差思想）
        out = clone_x + out
        return out


# 定义编码器
# 编码器层
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead() # 多头注意力计算层
        self.fc = FullyConnectedOutput() # 全连接输出层
    def forward(self, x, mask):
        # 计算自注意力，维度不变
        # [b,50,32]->[b,50,32]
        score = self.mh(x, x, x, mask) # Q=K=V
        # 全连接输出，维度不变
        # [b,50,32]->[b,50,32]
        out = self.fc(score)
        return out
# 编码器
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_l = EncoderLayer() # 编码器层
        self.layer_2 = EncoderLayer() # 编码器层
        self.layer_3 = EncoderLayer() # 编码器层
    def forward(self, x, mask):
        x = self.layer_l(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


# 定义解码器
# 解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mhl = MultiHead() # 多头注意力计算层
        self.mh2 = MultiHead() # 多头注意力计算层
        self.fc = FullyConnectedOutput() # 全连接输出层
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力，维度不变
        # [b,50,32] -> [b,50,32]
        y = self.mhl(y, y, y, mask_tril_y) # Q=K=V，其中Q、K和V表示编码器的输入，mask_tril_y表示[上三角矩阵 mask] + [PAD mask]
        # 结合x和y的注意力计算，维度不变
        # [b,50,32],[b,50,32]->[b,50,32]
        y = self.mh2(y, x, x, mask_pad_x) # Q=y, K=x, V=x。其中Q表示编码器的输出，K和V表示编码器的输入
        # 全连接输出，维度不变
        # [b,50,32]->[b,50,32]
        y = self.fc(y)
        return y

# 解码器
class Decoder(torch.nn.Module) :
    def __init__(self):
        super().__init__()
        self.layer_1 = DecoderLayer() # 解码器层
        self.layer_2 = DecoderLayer() # 解码器层
        self.layer_3 = DecoderLayer() # 解码器层
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y


# 定义主模型
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding() # 位置编码层
        self.embed_y = PositionEmbedding() # 位置编码层
        self.encoder = Encoder() # 编码器
        self.decoder = Decoder() # 解码器
        self.fc_out = torch.nn.Linear(32, 39) # 全连接输出层
    def forward(self, x, y):
        # [b,1,50,50]
        mask_pad_x = mask_pad(x) # PAD mask
        mask_tril_y = mask_tril(y) # 上三角 mask
        # 编码，添加位置信息
        # x=[b,50]->[b,50,32]
        # y=[b,50]->[b,50,32]
        x, y =self.embed_x(x), self.embed_y(y)
        # 编码层计算
        # [b,50,32]->[b,50,32]
        x = self.encoder(x, mask_pad_x)
        # 解码层计算
        # [b,50,32],[b,50,32]->[b,50,32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        # 全连接输出，维度不变
        # [b,50,32]->[b,50,39]
        y = self.fc_out(y)
        return y


# 定义预测函数
def predict(x):
    # x=[1,50]
    model.eval()
    # [1,1,50,50]
    mask_pad_x = mask_pad(x)
    # 初始化输出，这个是固定值
    # [1,50]
    # [[0,2,2,2...]]
    target = [vocab_y['<SOS>']] + [vocab_y['<PAD>']] * 49 # 初始化输出，这个是固定值
    target = torch.LongTensor(target).unsqueeze(0) # 增加一个维度，shape变为[1,50]
    # x编码，添加位置信息
    # [1,50] -> [1,50,32]
    x = model.embed_x(x)
    # 编码层计算，维度不变
    # [1,50,32] -> [1,50,32]
    x = model.encoder(x, mask_pad_x)
    # 遍历生成第1个词到第49个词
    for i in range(49):
        # [1,50]
        y = target
        # [1, 1, 50, 50]
        mask_tril_y = mask_tril(y) # 上三角 mask
        # y编码，添加位置信息
        # [1, 50] -> [1, 50, 32]
        y = model.embed_y(y)
        # 解码层计算，维度不变
        # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        # 全连接输出，39分类
        #[1,50,32]-> [1,50,39]
        out = model.fc_out(y)
        # 取出当前词的输出
        # [1,50,39]->[1,39]
        out = out[:,i,:]
        # 取出分类结果
        # [1,39]->[1]
        out = out.argmax(dim=1).detach()
        # 以当前词预测下一个词，填到结果中
        target[:,i + 1] = out
        return target


# 定义训练函数
def train():
    loss_func = torch.nn.CrossEntropyLoss() # 定义交叉熵损失函数
    optim = torch.optim.Adam(model.parameters(), lr=2e-3) # 定义优化器
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5) # 定义学习率衰减策略
    for epoch in range(1):
        for i, (x, y) in enumerate(loader):
            # x=[8,50]
            # y=[8,51]
            # 在训练时用y的每个字符作为输入，预测下一个字符，所以不需要最后一个字
            # [8,50,39]
            pred = model(x, y[:, :-1]) # 前向计算
            # [8,50,39] -> [400,39]
            pred = pred.reshape(-1, 39) # 转形状
            # [8,51]->[400]
            y = y[:, 1:].reshape(-1) # 转形状
            # 忽略PAD
            select = y != vocab_y['<PAD>']
            pred = pred[select]
            y = y[select]
            loss = loss_func(pred, y) # 计算损失
            optim.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optim.step() # 更新参数
            if i % 20 == 0:
                # [select,39] -> [select]
                pred = pred.argmax(1) # 取出分类结果
                correct = (pred == y).sum().item() # 计算正确个数
                accuracy = correct / len(pred) # 计算正确率
                lr = optim.param_groups[0]['lr'] # 取出当前学习率
                print(epoch, i, lr, loss.item(), accuracy) # 打印结果，分别为：当前epoch、当前batch、当前学习率、当前损失、当前正确率
        sched.step() # 更新学习率


# 测试
def model_test():
    for i,(x, y) in enumerate(loader):
        break
    for i in range(8):
        print(i)
        print(''.join([vocab_xr[i] for i in x[i].tolist()])) # 将编码转换成字符
        print(''.join([vocab_yr[i] for i in y[i].tolist()])) # 将编码转换成字符
        print(''.join([vocab_yr[i] for i in predict(x[i].unsqueeze(0))[0].tolist()])) # 将编码转换成字符


if __name__ == '__main__':
    # 测试get_data函数
    # x, y = get_data()

    # 数据集加载器
    # batch_size=8表示每次取8个样本
    # drop_last=True表示如果最后一个batch不够batch_size就丢弃
    # shuffle=True表示打乱数据
    # collate_fn=None表示使用默认的方式拼接数据
    loader = torch.utils.data.DataLoader(dataset=Dataset(), batch_size=8, drop_last=True, shuffle=True, collate_fn=None)
    # 查看数据样例
    for i, (x, y) in enumerate(loader):
        break
    # print(x.shape, y.shape)

    # 测试mask_pad函数
    # print(mask_pad(x[:1]))

    # 测试mask_tril函数
    # print(mask_tril(x[:1]))

    # 测试attention函数
    # print(attention(torch.randn(8, 4, 50, 8), torch.randn(8, 4, 50, 8), torch.randn(8, 4, 50, 8), torch.zeros(8, 1, 50, 50)).shape)

    # BatchNorm1d和LayerNorm的对比
    # 标准化之后，均值是0, 标准差是1
    # BN是取不同样本做标准化
    # LN是取不同通道做标准化
    # affine=True,elementwise_affine=True：指定标准化后再计算一个线性映射
    # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
    # norm_data = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    # result_norm = norm(norm_data)
    # print(result_norm)
    # norm = torch.nn.LayerNorm(normalized_shape=4, elementwise_affine=True)
    # norm_data = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    # result_norm = norm(norm_data)
    # print(result_norm)

    # 测试MultiHead函数
    # print(MultiHead()(torch.randn(8, 50, 32), torch.randn(8, 50, 32), torch.randn(8, 50, 32), torch.zeros(8, 1, 50, 50)).shape)

    # 测试PositionEmbedding函数
    # PositionEmbedding_Result = PositionEmbedding()(torch.ones(8, 50).long())
    # print(PositionEmbedding_Result.shape)

    # 测试FullyConnectedOutput函数
    # print(FullyConnectedOutput()(torch.randn(8, 50, 32)).shape)

    # 测试EncoderLayer函数
    # print(Encoder()(torch.randn(8, 50, 32), torch.ones(8, 1, 50, 50)))

    # 测试Decoder函数
    # print(Decoder()(torch.randn(8, 50, 32), torch.randn(8, 50, 32), torch.ones(8, 1, 50, 50), torch.ones(8, 1, 50, 50)).shape)

    # 测试Transformer函数
    model = Transformer()
    # print(model(torch.ones(8, 50).long(), torch.ones(8, 50).long()).shape)

    # 测试train函数
    train()

    # 测试predict函数
    print(predict(torch.ones(1, 50).long()))

    # 测试test函数
    # model_test()