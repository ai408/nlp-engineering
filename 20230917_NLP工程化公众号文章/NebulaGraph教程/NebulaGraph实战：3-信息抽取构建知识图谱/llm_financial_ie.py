#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 17:03
# @Author  : 陈卓见
# @File    : llm_financial_ie.py
# @Description : 这个函数是用来干llm_financial_ie的
# !/usr/bin/env python3

import re
import json

from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel


# 分类 example
class_examples = {
        '基金': '4月21日，易方达基金公司明星基金经理张坤在管的4只基金产品悉数发布2023年一季报。通联数据显示，一季度张坤的在管规模为889.42亿元，虽然较2022年末的894.34亿元减少约4.92亿元，但同比2022年一季度末的849.27亿元增加了39.15亿元。一季度，除了易方达优质企业三年持有处于封闭期内，其余三只基金产品均遭到了净赎回。其中，易方达蓝筹精选的净赎回份额为6.64亿份。受此影响，3月末张坤的总管理规模较去年年末小幅降低，为889.42亿元。',
        '股票': '国联证券04月23日发布研报称，给予东方财富（300059.SZ，最新价：17.03元）买入评级，目标价格为22.12元。董事长其实表示，对于东方财富未来的增长充满信心'
    }
class_list = list(class_examples.keys()) # 获取分类的类别，class_list = ['基金', '股票']

CLS_PATTERN = f"“{{}}”是 {class_list} 里的什么类别？"


# 定义不同实体下的具备属性
schema = {
    '基金': ['基金名称', '基金经理', '基金公司', '基金规模', '重仓股'],
    '股票': ['股票名称', '董事长', '涨跌幅']
}

IE_PATTERN = "{}\n\n提取上述句子中{}类型的实体，并按照JSON格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','分隔。"


# 提供一些例子供模型参考
ie_examples = {
        '基金': [
                    {
                        'content': '4月21日，易方达基金公司明星基金经理张坤在管的4只基金产品悉数发布2023年一季报。通联数据显示，一季度张坤的在管规模为889.42亿元，虽然较2022年末的894.34亿元减少约4.92亿元，但同比2022年一季度末的849.27亿元增加了39.15亿元。一季度，除了易方达优质企业三年持有处于封闭期内，其余三只基金产品均遭到了净赎回。其中，易方达蓝筹精选的净赎回份额为6.64亿份。受此影响，3月末张坤的总管理规模较去年年末小幅降低，为889.42亿元。重仓茅台和五粮液。',
                        'answers': {
                                        '基金名称': ['易方达优质企业','易方达蓝筹精选'],
                                        '基金经理': ['张坤'],
                                        '基金公司': ['易方达基金公司'],
                                        '基金规模': ['889.42亿元'],
                                        '重仓股': ['五浪液','茅台']
                            }
                    }
        ],
        '股票': [
                    {
                        'content': '国联证券04月23日发布研报称，给予东方财富（300059.SZ，最新价：17.03元）买入评级，目标价格为22.12元。董事长其实表示，对于东方财富未来的增长充满信心',
                        'answers': {
                                        '股票名称': ['东方财富'],
                                        '董事长': ['其实'],
                                        '涨跌幅': ['原文中未提及']
                            }
                    }
        ]
}


def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    class_list = list(class_examples.keys()) # 获取分类的类别，class_list = ['基金', '股票']
    cls_pre_history = [
        (
            f'现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：{class_list}类别中。',
            f'好的。'
        )
    ]

    for _type, exmpale in class_examples.items(): # 遍历分类的类别和例子
        cls_pre_history.append((f'“{exmpale}”是 {class_list} 里的什么类别？', _type)) # 拼接前置prompt

    ie_pre_history = [
        (
            "现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中三元组，并按照JSON的格式输出，上述句子中没有的信息用['原文中未提及']来表示，多个值之间用','分隔。",
            '好的，请输入您的句子。'
        )
    ]

    for _type, example_list in ie_examples.items(): # 遍历分类的类别和例子
        for example in example_list: # 遍历例子
            sentence = example['content'] # 获取句子
            properties_str = ', '.join(schema[_type]) # 拼接schema
            schema_str_list = f'“{_type}”({properties_str})' # 拼接schema
            sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list) # 拼接前置prompt
            ie_pre_history.append(( # 拼接前置prompt
                f'{sentence_with_prompt}',
                f"{json.dumps(example['answers'], ensure_ascii=False)}"
            ))

    return {'ie_pre_history': ie_pre_history, 'cls_pre_history': cls_pre_history} # 返回前置prompt


def clean_response(response: str):
    """
    后处理模型输出。

    Args:
        response (str): _description_
    """
    if '```json' in response: # 如果是json格式的，需要提取出来
        res = re.findall(r'```json(.*?)```', response)  # 提取json格式的内容
        if len(res) and res[0]: # 如果有内容
            response = res[0] # 提取出来
        response.replace('、', ',')
    try:
        return json.loads(response) # 尝试转换为json格式
    except:
        return response


def inference(
        sentences: list,
        custom_settings: dict
    ):
    """
    推理函数。

    Args:
        sentences (List[str]): 待抽取的句子。
        custom_settings (dict): 初始设定，包含人为给定的few-shot example。
    """
    for sentence in sentences: # 遍历句子
        with console.status("[bold bright_green] Model Inference..."): # 显示推理中
            sentence_with_cls_prompt = CLS_PATTERN.format(sentence) # 拼接前置prompt
            cls_res, _ = model.chat(tokenizer, sentence_with_cls_prompt, history=custom_settings['cls_pre_history']) # 推理

            if cls_res not in schema: # 如果推理结果不在schema中，报错并退出
                print(f'The type model inferenced {cls_res} which is not in schema dict, exited.')
                exit()

            properties_str = ', '.join(schema[cls_res]) # 拼接schema
            schema_str_list = f'“{cls_res}”({properties_str})' # 拼接schema
            sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list) # 拼接前置prompt
            ie_res, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=custom_settings['ie_pre_history']) # 推理
            ie_res = clean_response(ie_res) # 后处理
        print(f'>>> [bold bright_red]sentence: {sentence}') # 打印句子
        print(f'>>> [bold bright_green]inference answer: ') # 打印推理结果
        print(ie_res) # 打印推理结果


if __name__ == '__main__':
    console = Console() # 初始化console，用于显示信息

    device = 'cuda:0' # 指定使用GPU
    tokenizer = AutoTokenizer.from_pretrained(r"L:/20230713_HuggingFaceModel/chatglm2-6b", trust_remote_code=True) # 指定使用的tokenizer
    model = AutoModel.from_pretrained(r"L:/20230713_HuggingFaceModel/chatglm2-6b", trust_remote_code=True).half().cuda() # 指定使用的model
    model = model.eval() # 指定model为eval模式

    sentences = [ # 待抽取的句子
        '同花顺董事长易峥虚拟“分身”谈人工智能：ALL IN但也要保持清醒',
        '葛兰一季度末管理规模再度跌破900亿元，同时交出了公募主动权益基金管理规模的头把交椅。从管理规模来看，一季度末葛兰在管5只公募基金合计管理规模降至844.40亿元，较2022年末的906.53亿元，环比下降6.85%。葛兰一季度调仓力度并不大。以中欧医疗健康混合基金为例',
        '东财芯片ETF即将发售，初始募集金额为20亿'
    ]

    custom_settings = init_prompts() # 初始化前置prompt
    inference( # 推理
        sentences,
        custom_settings
    )