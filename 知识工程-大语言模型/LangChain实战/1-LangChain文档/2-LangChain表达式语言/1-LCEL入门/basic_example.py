from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from api_secret_key import API_Key, API_SECRET
import os
os.environ["QIANFAN_AK"] = API_Key
os.environ["QIANFAN_SK"] = API_SECRET


# 使用文心一言模型
def chat_with_qianfan() -> QianfanChatEndpoint:
    """For basic init and call"""

    chat = QianfanChatEndpoint(streaming=True,  # 是否使用流式模式
                               model="ERNIE-Bot-turbo",  # 使用的模型
                               **{"top_p": 0.8, "temperature": 0.95, "penalty_score": 1}  # 模型参数
    )
    return chat


def basic_demo(model: QianfanChatEndpoint):
    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")  # 给我讲一个关于{topic}的笑话
    output_parser = StrOutputParser()  # 使用字符串输出解析器
    chain = prompt | model | output_parser  # 创建链条
    chain_response = chain.invoke({"topic": "ice cream"})  # 运行链条
    print(chain_response)  # 打印结果

    # 1.提示
    prompt_value = prompt.invoke({"topic": "ice cream"})
    print(prompt_value)  # 打印结果
    prompt_value_message = prompt_value.to_messages()  # 打印消息
    print(prompt_value_message)  # 打印消息

    # 2.模型
    message = model.invoke(prompt_value)
    print(message)  # 打印结果

    # 3.输出解析器
    output_parser_response = output_parser.invoke(message)
    print(output_parser_response)  # 打印结果

    # 4.整个流程
    input = {"topic": "ice cream"}
    prompt_response = prompt.invoke(input)
    print(prompt_response)  # 打印结果
    prompt_model_response = (prompt | model).invoke(input)
    print(prompt_model_response)  # 打印结果


if __name__ == '__main__':
    model = chat_with_qianfan()
    basic_demo(model)
