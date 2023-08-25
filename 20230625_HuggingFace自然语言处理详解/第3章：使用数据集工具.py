#第3章/加载数据集
from datasets import load_dataset
dataset = load_dataset(path='seamew/ChnSentiCorp')
print(dataset)


# dataset = load_dataset(path='seamew/ChnSentiCorp', name='sst2', split='train')
# print(dataset)


#第3章/使用批处理加速
# def f(data):
#     text=data['text']
#     text=['My sentence: ' + i for i in text]
#     data['text']=text
#     return data
# maped_datatset=dataset.map(function=f, batched=True, batch_size=1000, num_proc=4)
# print(dataset['text'][20])
# print(maped_datatset['text'][20])


#第3章/设置数据格式
# dataset.set_format(type='torch', columns=['label'], output_all_columns=True)
# print(dataset[20])


#第3章/导出为CSV格式
# dataset = load_dataset(path='seamew/ChnSentiCorp', split='train')
# dataset.to_csv(path_or_buf='./data/ChnSentiCorp.csv')
# #加载CSV格式数据
# csv_dataset = load_dataset(path='csv', data_files='./data/ChnSentiCorp.csv', split='train')
# print(csv_dataset[20])




if __name__ == '__main__':
    pass