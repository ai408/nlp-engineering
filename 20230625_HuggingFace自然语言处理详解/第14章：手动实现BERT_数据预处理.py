# 数据处理实现代码


# 1.字词处理
# 读取数据文件
import pandas as pd
# 设置分隔符为制表符和换行符
data = pd.read_csv('dataset/msr-para-train.tsv', sep='\t|\n')

# 删除无用的两列数据
data.pop('#1 ID')
data.pop('#2 ID')

# 重命名列
columns = list(data.columns)
columns[0] = 'same'
columns[1] = 's1'
columns[2] = 's2'
data.columns = columns # 重命名列

# 删除文本中的<QUOTE>符号
data['s1'] = data['s1'].str.replace('<QUOTE>', ' ')
data['s2'] = data['s2'].str.replace('<QUOTE>', ' ')

# 删除标点符号
data['s1'] = data['s1'].str.replace('[^\w\s]', ' ')
data['s2'] = data['s2'].str.replace('[^\w\s]', ' ')

# 替换特殊字符
data['s1'] = data['s1'].str.replace('â', 'a')
data['s2'] = data['s2'].str.replace('â', 'a')
data['s1'] = data['s1'].str.replace('Â', 'A')
data['s2'] = data['s2'].str.replace('Â', 'A')
data['s1'] = data['s1'].str.replace('Ã', 'A')
data['s2'] = data['s2'].str.replace('Ã', 'A')
data['s1'] = data['s1'].str.replace(' ', ' ')
data['s2'] = data['s2'].str.replace(' ', ' ')
data['s1'] = data['s1'].str.replace('μ', 'u')
data['s2'] = data['s2'].str.replace('μ', 'u')
data['s1'] = data['s1'].str.replace('³', ' ')
data['s2'] = data['s2'].str.replace('³', ' ')
data['s1'] = data['s1'].str.replace('½', ' ')
data['s2'] = data['s2'].str.replace('½', ' ')

# 合并连续的空格
data['s1'] = data['s1'].str.replace('\s{2,}', ' ')
data['s2'] = data['s2'].str.replace('\s{2,}', ' ')

# 拆分数字和字母连写的词
data['s1'] = data['s1'].str.replace('(\d)([a-zA-Z])', '\\1 \\2')
data['s2'] = data['s2'].str.replace('(\d)([a-zA-Z])', '\\1 \\2')
data['s1'] = data['s1'].str.replace('([a-zA-Z])(\d)', '\\1 \\2')
data['s2'] = data['s2'].str.replace('([a-zA-Z])(\d)', '\\1 \\2')

# 删除首尾空格并小写所有字母
data['s1'] = data['s1'].str.strip()
data['s2'] = data['s2'].str.strip()
data['s1'] = data['s1'].str.lower()
data['s2'] = data['s2'].str.lower()

# 替换数字为符号
data['s1'] = data['s1'].str.replace('\d+', '<NUM>')
data['s2'] = data['s2'].str.replace('\d+', '<NUM>')


# 2.合并句子
# 为s1添加首尾符号
def f(sent):
    return '<SOS> ' + sent + ' <EOS>'
data['s1'] = data['s1'].apply(f)

# 为s2添加结尾符号
def f(sent):
    return sent + ' <EOS>'
data['s2'] = data['s2'].apply(f)

# 分别求出s1和s2的长度
def f(sent):
    return len(sent.split(' '))
data['s1_lens'] = data['s1'].apply(f)
data['s2_lens'] = data['s2'].apply(f)

# 求s1+s2后的最大长度
max_lens = max(data['s1_lens'] + data['s2_lens'])

# 求出每个句子需要补充PAD的长度
data['pad_lens'] = max_lens - data['s1_lens'] - data['s2_lens']

# 合并s1和s2
data['sent'] = data['s1'] + ' ' + data['s2']
data.pop('s1')
data.pop('s2')

# 为不足最大长度的句子补充PAD
def f(row):
    pad = ' '.join(['<PAD>'] * row['pad_lens'])
    row['sent'] = row['sent'] + ' ' + pad
    return row
data = data.apply(f, axis=1)


# 3.构建字典并编码
# 构建字典
def build_vocab():
    vocab = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<NUM>': 3,
    '<UNK>': 4,
    '<MASK>': 5,
    '<Symbol6>': 6,
    '<Symbol7>': 7,
    '<Symbol8>': 8,
    '<Symbol9>': 9,
    '<Symbol10>': 10,
    }
    for i in range(len(data)):
        for word in data.iloc[i]['sent'].split(' '):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab
vocab = build_vocab()
print(len(vocab), vocab['the'])

# 使用字典编码文本
def f(sent):
    sent = [str(vocab[word]) for word in sent.split()]
    sent = ','.join(sent)
    return sent
data['sent'] = data['sent'].apply(f)


# 4.保存数据文件
# 保存为CSV文件
data.to_csv('dataset/msr_paraphrase_data_train.csv', index=False)

# 保存字典
pd.DataFrame(vocab.items(), columns=['word', 'token']).to_csv('dataset/msr_paraphrase_vocab.csv', index=False)