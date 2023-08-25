def encode_example_test():
    # 字典
    vocab = {
        '<SOS>': 0,
        '<EOS>': 1,
        'the': 2,
        'quick': 3,
        'brown': 4,
        'fox': 5,
        'jumps': 6,
        'over': 7,
        'a': 8,
        'lazy': 9,
        'dog': 10,
    }

    # 简单编码
    sent = 'the quick brown fox jumps over a lazy dog'
    sent = '<SOS> ' + sent + ' <EOS>'
    print(sent)

    # 英文分词
    words = sent.split()
    print(words)

    # 编码为数字
    encode = [vocab[i] for i in words]
    print(encode)


def encode_test():
    # 第2章/加载编码工具
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',  # 通常编码工具和模型名字一致
        cache_dir=None,  # 编码工具的缓存路径
        force_download=False,  # 是否强制下载，当为True时，无论是否有本地缓存，都会强制下载
    )

    # 第2章/准备实验数据
    sents = [
        '你站在桥上看风景',
        '看风景的人在楼上看你',
        '明月装饰了你的窗子',
        '你装饰了别人的梦',
    ]

    # 第2章/基本的编码函数
    out = tokenizer.encode(
        text=sents[0],
        text_pair=sents[1],  # 如果只想编码一个句子，可设置text_pair=None
        truncation=True,  # 当句子长度大于max_length时截断
        padding='max_length',  # 一律补PAD，直到max_length长度
        add_special_tokens=True,  # 需要在句子中添加特殊符号
        max_length=25,  # 最大长度
        return_tensors=None,  # 返回的数据类型为list格式，也可以赋值为tf、pt、np，分别表示TensorFlow、PyTorch、NumPy数据格式
    )
    print(out)
    print(tokenizer.decode(out))


def encode_plus_test():
    # 第2章/加载编码工具
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',  # 通常编码工具和模型名字一致
        cache_dir=None,  # 编码工具的缓存路径
        force_download=False,  # 是否强制下载，当为True时，无论是否有本地缓存，都会强制下载
    )

    # 第2章/准备实验数据
    sents = [
        '你站在桥上看风景',
        '看风景的人在楼上看你',
        '明月装饰了你的窗子',
        '你装饰了别人的梦',
    ]

    # 第2章/进阶的编码函数
    out = tokenizer.encode_plus(
        text=sents[0],
        text_pair=sents[1],
        truncation=True,  # 当句子长度大于max_length时截断
        padding='max_length',  # 一律补零，直到max_length长度
        max_length=25,
        add_special_tokens=True,
        return_tensors=None,  # 可取值tf、pt、np，默认为返回list
        return_token_type_ids=True,  # 返回token_type_ids：第1个句子和特殊符号的位置是0，第2个句子的位置是1
        return_attention_mask=True,  # 返回attention_mask：PAD的位置是0，其他位置是1
        return_special_tokens_mask=True,  # 返回special_tokens_mask特殊符号标识：特殊符号的位置是1，其他位置是0
        return_length=True,  # 返回编码后句子的长度
    )

    # input_ids：编码后的词
    # token_type_ids：第1个句子和特殊符号的位置是0，第2个句子的位置是1
    # special_tokens_mask：特殊符号的位置是1，其他位置是0
    # attention_mask：PAD的位置是0，其他位置是1
    # length：返回句子长度
    for k, v in out.items():
        print(k, ':', v)
    print(tokenizer.decode(out['input_ids']))


def batch_encode_plus_test():
    # 第2章/加载编码工具
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',  # 通常编码工具和模型名字一致
        cache_dir=None,  # 编码工具的缓存路径
        force_download=False,  # 是否强制下载，当为True时，无论是否有本地缓存，都会强制下载
    )

    # 第2章/准备实验数据
    sents = [
        '你站在桥上看风景',
        '看风景的人在楼上看你',
        '明月装饰了你的窗子',
        '你装饰了别人的梦',
    ]

    # 第2章/批量编码成对的句子
    out = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
        # 编码成对的句子，如果只想编码一个句子，那么batch_text_or_text_pairs=[sents[0], sents[1]]
        add_special_tokens=True,  # 需要在句子中添加特殊符号
        truncation=True,  # 当句子长度大于max_length时截断
        padding='max_length',  # 一律补零，直到max_length长度
        max_length=25,
        return_tensors=None,  # 可取值tf、pt、np，默认为返回list
        return_token_type_ids=True,  # 返回token_type_ids：第1个句子和特殊符号的位置是0，第2个句子的位置是1
        return_attention_mask=True,  # 返回attention_mask：PAD的位置是0，其他位置是1
        return_special_tokens_mask=True,  # 返回special_tokens_mask特殊符号标识：特殊符号的位置是1，其他位置是0
        # return_offsets_mapping=True, # 返回offsets_mapping标识每个词的起止位置，这个参数只能BertTokenizerFast使用
        return_length=True,  # 返回编码后句子的长度
    )
    # input_ids：编码后的词
    # token_type_ids：第1个句子和特殊符号的位置是0，第2个句子的位置是1
    # special_tokens_mask：特殊符号的位置是1，其他位置是0
    # attention_mask：PAD的位置是0，其他位置是1
    # length：返回句子长度
    for k, v in out.items():
        print(k, ':', v)
    tokenizer.decode(out['input_ids'][0])


def dict_test():
    # 第2章/加载编码工具
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',  # 通常编码工具和模型名字一致
        cache_dir=None,  # 编码工具的缓存路径
        force_download=False,  # 是否强制下载，当为True时，无论是否有本地缓存，都会强制下载
    )

    # 第2章/获取字典
    vocab = tokenizer.get_vocab()
    print(type(vocab), len(vocab), '明月' in vocab)  # <class 'dict'> 21128 False

    # 第2章/添加新词
    tokenizer.add_tokens(new_tokens=['明月', '装饰', '窗子'])

    # 第2章/添加新符号
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    # 第2章/编码新添加的词
    out = tokenizer.encode(
        text='明月装饰了你的窗子[EOS]',
        text_pair=None,
        truncation=True,  # 当句子长度大于max_length时截断
        padding='max_length',  # 一律补PAD，直到max_length长度
        add_special_tokens=True,  # 需要在句子中添加特殊符号
        max_length=10,
        return_tensors=None,  # 可取值tf、pt、np，默认为返回list
    )
    print(out)
    print(tokenizer.decode(out))  # [CLS] 明月 装饰 了 你 的 窗子 [EOS] [SEP] [PAD]


if __name__ == "__main__":
    # encode_example_test()

    # encode_test()

    # encode_plus_test()

    # batch_encode_plus_test()

    dict_test()
