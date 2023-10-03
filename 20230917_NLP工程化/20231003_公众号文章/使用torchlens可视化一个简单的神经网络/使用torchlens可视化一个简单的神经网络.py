import torch
import torch.nn as nn
import torch.optim as optim
import torchlens as tl
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin/'


# 定义神经网络类
class NeuralNetwork(nn.Module): # 继承nn.Module类
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__() # 调用父类的构造函数
        # 定义输入层到隐藏层的线性变换
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        # 定义隐藏层到输出层的线性变换
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        # 定义激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 前向传播
        hidden = self.sigmoid(self.input_to_hidden(x))
        output = self.sigmoid(self.hidden_to_output(hidden))
        return output

def NeuralNetwork_train(model):
    # 训练神经网络
    for epoch in range(10000):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(input_data)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播和优化
        optimizer.step()  # 更新参数

        # 每100个epoch打印一次损失
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}')

    return model


def NeuralNetwork_test(model):
    # 在训练后，可以使用模型进行预测
    with torch.no_grad():
        test_input = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        predictions = model(test_input)
        predicted_labels = (predictions > 0.5).float()
        print("Predictions:", predicted_labels)


if __name__ == '__main__':
    # 定义神经网络的参数
    input_size = 2  # 输入特征数量
    hidden_size = 4  # 隐藏层神经元数量
    output_size = 1  # 输出层神经元数量

    # 创建神经网络实例
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

    # 准备示例输入数据和标签
    input_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # model：神经网络模型
    # input_data：输入数据
    # layers_to_save：需要保存的层
    # vis_opt：rolled/unrolled，是否展开循环
    model_history = tl.log_forward_pass(model, input_data, layers_to_save='all', vis_opt='unrolled')  # 可视化神经网络
    print(model_history)
    # print(model_history['input_1'])
    # print(model_history['input_1'].tensor_contents)

    tl.show_model_graph(model, input_data) # 可视化神经网络

    model = NeuralNetwork_train(model) # 训练神经网络
    # NeuralNetwork_test(model) # 测试神经网络