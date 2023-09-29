# 1.定义测试数据
# 虚拟数据
import torch
# 假设有两句话，8个词
x = torch.ones(2, 8)
# 两句话中各有一些PAD
x[0, 6:] = 0
x[1, 7:] = 0
# print(x) # x的shape是[2, 8]


# 2.各个MASK的含义解释
# 定义key_padding_mask
# key_padding_mask的定义方式，就是x中是pad的为True，否则是False
key_padding_mask = x == 0
# print(key_padding_mask)

# 定义encode_attn_mask
# 在encode阶段不需要定义encode_attn_mask
# 定义为None或者全False都可以
encode_attn_mask = torch.ones(8, 8) == 0
# print(encode_attn_mask)

# 定义decode_attn_mask
# 在decode阶段需要定义decode_attn_mask
# decode_attn_mask的定义方式是对角线以上为True的上三角矩阵
decode_attn_mask = torch.tril(torch.ones(8, 8)) == 0
# print(decode_attn_mask)


# 3.编码数据
# 编码x
x = x.unsqueeze(2) # 在第2维增加一个维度
x = x.expand(-1, -1, 12) # 在第2维复制12份，扩展为[2, 8, 12]
# print(x, x.shape)


# 4.多头注意力计算函数
# 定义multi_head_attention_forward()所需要的参数
# in_proj就是Q、K、V线性变换的参数
in_proj_weight = torch.nn.Parameter(torch.randn(3 * 12, 12))
in_proj_bias = torch.nn.Parameter(torch.zeros((3 * 12)))
# out_proj就是输出时做线性变换的参数
out_proj_weight = torch.nn.Parameter(torch.randn(12, 12))
out_proj_bias = torch.nn.Parameter(torch.zeros(12))
# print(in_proj_weight.shape, in_proj_bias.shape)
# print(out_proj_weight.shape, out_proj_bias.shape)

# 使用工具函数计算多头注意力
data = {
    # 因为不是batch_first的，所以需要进行变形
    'query': x.permute(1, 0, 2), # x原始为[2, 8, 12]，x.permute为[8, 2, 12]
    'key': x.permute(1, 0, 2),
    'value': x.permute(1, 0, 2),
    'embed_dim_to_check': 12, # 用于检查维度是否正确
    'num_heads': 2, # 多头注意力的头数
    'in_proj_weight': in_proj_weight, # Q、K、V线性变换的参数
    'in_proj_bias': in_proj_bias, # Q、K、V线性变换的参数
    'bias_k': None,
    'bias_v': None,
    'add_zero_attn': False,
    'dropout_p': 0.2, # dropout的概率
    'out_proj_weight': out_proj_weight, # 输出时做线性变换的参数
    'out_proj_bias': out_proj_bias, # 输出时做线性变换的参数
    'key_padding_mask': key_padding_mask,
    'attn_mask': encode_attn_mask,
}
score, attn = torch.nn.functional.multi_head_attention_forward(**data)
# print(score.shape, attn, attn.shape)


# 5.多头注意力层
# 使用多头注意力工具层
multihead_attention = torch.nn.MultiheadAttention(embed_dim=12, num_heads=2, dropout=0.2, batch_first=True)
data = {
    'query': x,
    'key': x,
    'value': x,
    'key_padding_mask': key_padding_mask,
    'attn_mask': encode_attn_mask,
}
score, attn = multihead_attention(**data)
# print(score.shape, attn, attn.shape)


# 6.编码器层
# 使用单层编码器工具层
encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=12,                          # 词向量的维度
                        nhead=2,                             # 多头注意力的头数
                        dim_feedforward=24,                  # 前馈神经网络的隐层维度
                        dropout=0.2,                         # dropout的概率
                        activation=torch.nn.functional.relu, # 激活函数
                        batch_first=True,                    # 输入数据的第一维是batch
                        norm_first=True)                     # 归一化层在前
data = {
    'src': x,                                 # 输入数据
    'src_mask': encode_attn_mask,             # 输入数据的mask
    'src_key_padding_mask': key_padding_mask, # 输入数据的key_padding_mask
}
out = encoder_layer(**data)
# print(out.shape)

# 使用编码器工具层
encoder = torch.nn.TransformerEncoder(
    encoder_layer=encoder_layer,                  # 编码器层
    num_layers=3,                                 # 编码器层数
    norm=torch.nn.LayerNorm(normalized_shape=12)) # 归一化层
data = {
    'src': x, # 输入数据
    'mask': encode_attn_mask,                     # 输入数据的mask
    'src_key_padding_mask': key_padding_mask,     # 输入数据的key_padding_mask
}
out = encoder(**data)
# print(out.shape)


# 7.解码器层
#  使用单层解码器工具层
decoder_layer = torch.nn.TransformerDecoderLayer(    # 解码器层
                d_model=12,                          # 词向量的维度
                nhead=2,                             # 多头注意力的头数
                dim_feedforward=24,                  # 前馈神经网络的隐层维度
                dropout=0.2,                         # dropout的概率
                activation=torch.nn.functional.relu, # 激活函数
                batch_first=True,                    # 输入数据的第一维是batch
                norm_first=True)                     # 归一化层在前
data = {
    'tgt': x,                                        # 解码输出的目标语句，即target
    'memory': x,                                     # 编码器的编码结果，即解码器解码时的根据数据
    'tgt_mask': decode_attn_mask,                    # 定义是否要忽略词与词之间的注意力，即decode_attn_mask
    'memory_mask': encode_attn_mask,                 # 定义是否要忽略memory内的部分词与词之间的注意力，一般不需要要忽略
    'tgt_key_padding_mask': key_padding_mask,        # 定义target内哪些位置是PAD，以忽略对PAD的注意力
    'memory_key_padding_mask': key_padding_mask,     # 定义memory内哪些位置是PAD，以忽略对PAD的注意力
}
out = decoder_layer(**data)
# print(out.shape)

# 使用编码器工具层
decoder = torch.nn.TransformerDecoder(    # 解码器层
    decoder_layer=decoder_layer,          # 解码器层
    num_layers=3,                         # 解码器层数
    norm=torch.nn.LayerNorm(normalized_shape=12))
data = {
    'tgt': x,
    'memory': x,
    'tgt_mask': decode_attn_mask,
    'memory_mask': encode_attn_mask,
    'tgt_key_padding_mask': key_padding_mask,
    'memory_key_padding_mask': key_padding_mask,
}
out = decoder(**data)
# print(out.shape)


# 8.完整的Transformer模型
# 使用Transformer工具模型
transformer = torch.nn.Transformer(d_model=12,               # 词向量的维度
                        nhead=2,                             # 多头注意力的头数
                        num_encoder_layers=3,                # 编码器层数
                        num_decoder_layers=3,                # 解码器层数
                        dim_feedforward=24,                  # 前馈神经网络的隐层维度
                        dropout=0.2,                         # dropout的概率
                        activation=torch.nn.functional.relu, # 激活函数
                        custom_encoder=encoder,              # 自定义编码器，如果指定为None，那么会使用默认的编码器层堆叠num_encoder_layers层组成编码器
                        custom_decoder=decoder,              # 自定义解码器，如果指定为None，那么会使用默认的解码器层堆叠num_decoder_layers层组成解码器
                        batch_first=True,                    # 输入数据的第一维是batch
                        norm_first=True)                     # 归一化层在前
data = {
    'src': x,
    'tgt': x,
    'src_mask': encode_attn_mask,
    'tgt_mask': decode_attn_mask,
    'memory_mask': encode_attn_mask,
    'src_key_padding_mask': key_padding_mask,
    'tgt_key_padding_mask': key_padding_mask,
    'memory_key_padding_mask': key_padding_mask,
}
out = transformer(**data)
# print(out.shape)