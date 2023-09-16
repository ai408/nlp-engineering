"""
https://www.yii666.com/blog/395011.html
https://cloud.tencent.com/developer/article/2263711
https://mp.weixin.qq.com/s/2VuZOwe6rf3uAYyoXXPloQ
https://www.zhihu.com/question/371094177/answer/2999185806
https://zhuanlan.zhihu.com/p/643639728
大模型微调方法综述：http://www.guyuehome.com/43650
预训练模型发展——大模型进阶之路：https://zhuanlan.zhihu.com/p/606910992
ChatGPT复现之路：https://www.cnblogs.com/wangbin/p/17328802.html
大语言模型的预训练：https://cloud.tencent.com/developer/article/2303090
DeepSpeed Integration：https://huggingface.co/docs/transformers/main/main_classes/deepspeed
https://github.com/liucongg/ChatGPTBook
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import deepspeed


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True) # 下载数据集
trainloader = torch.utils.data.DataLoader(trainset,batch_size=16, shuffle=True, num_workers=2) # 定义数据加载器
testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=True) # 下载数据集
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2) # 定义数据加载器


class Net(nn.Module): # 定义网络
    def __init__(self): # 初始化网络
        super(Net, self).__init__() # 继承父类初始化
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义卷积层
        self.pool = nn.MaxPool2d(2, 2) # 定义池化层
        self.conv2 = nn.Conv2d(6, 16, 5) # 定义卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 定义全连接层
        self.fc2 = nn.Linear(120, 84) # 定义全连接层
        self.fc3 = nn.Linear(84, 10)  # 定义全连接层

    def forward(self, x): # 定义前向传播
        x = self.pool(F.relu(self.conv1(x))) # 卷积->激活->池化
        x = self.pool(F.relu(self.conv2(x))) # 卷积->激活->池化
        x = x.view(-1, 16 * 5 * 5) # 展平
        x = F.relu(self.fc1(x)) # 全连接->激活
        x = F.relu(self.fc2(x)) # 全连接->激活
        x = self.fc3(x) # 全连接
        return x # 返回输出


def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR') # 定义参数
    parser.add_argument('-b', # 定义batch_size
                        '--batch_size', # 参数名称
                        default=32, # 默认值
                        type=int, # 参数类型
                        help='mini-batch size (default: 32)') # 参数描述
    parser.add_argument('-e', # 定义epoch
                        '--epochs', # 参数名称
                        default=30, # 默认值
                        type=int, # 参数类型
                        help='number of total epochs (default: 30)') # 参数描述
    parser.add_argument('--local_rank', # 定义local_rank
                        type=int,   # 参数类型
                        default=-1, # 默认值
                        help='local rank passed from distributed launcher') # 参数描述

    parser.add_argument('--log-interval', # 定义log间隔
                        type=int, # 参数类型
                        default=2000, # 默认值
                        help="output logging information at a given interval") # 参数描述

    parser = deepspeed.add_config_arguments(parser) # 添加deepspeed参数
    args = parser.parse_args() # 解析参数
    return args # 返回参数



if __name__ == '__main__':
    net = Net() # 定义网络
    criterion = nn.CrossEntropyLoss() # 定义损失函数
    args = add_argument() # 定义参数
    parameters = filter(lambda p: p.requires_grad, net.parameters()) # 过滤出需要梯度更新的参数
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=net, model_parameters=parameters, training_data=trainset) # 初始化引擎

    # 训练逻辑
    for epoch in range(2): # 训练2个epoch
        running_loss = 0.0 # 初始化loss
        for i, data in enumerate(trainloader): # 遍历数据集
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank) # 获取数据
            outputs = model_engine(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算loss
            model_engine.backward(loss) # 反向传播
            model_engine.step() # 更新参数

            # 打印log信息
            running_loss += loss.item() # 累加loss
            if i % args.log_interval == (args.log_interval - 1): # 每隔log_interval打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.log_interval)) # 打印log信息
                running_loss = 0.0 # 重置loss

    # 测试逻辑
    correct = 0  # 定义正确数
    total = 0 # 定义总数
    with torch.no_grad(): # 关闭梯度更新
        for data in testloader: # 遍历数据集
            images, labels = data # 获取数据
            outputs = net(images.to(model_engine.local_rank)) # 前向传播
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果
            total += labels.size(0) # 累加总数
            correct += (predicted == labels.to(model_engine.local_rank)).sum().item() # 累加正确数
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total)) # 打印准确率