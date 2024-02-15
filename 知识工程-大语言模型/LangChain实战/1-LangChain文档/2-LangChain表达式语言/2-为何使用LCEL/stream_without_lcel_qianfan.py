from typing import List, Iterator
import qianfan
from langchain_community.chat_models import QianfanChatEndpoint

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

def stream_chat_model(messages: List[dict]) -> Iterator[str]:
    client = qianfan.ChatCompletion()  # 创建客户端
    stream = client.do(  # 调用模型
        model="ERNIE-Bot-turbo",
        messages=messages,
        stream=True,
    )
    for response in stream:  # 逐条处理stream
        content = response.body['result']  # 获取content
        if content is not None:  # 如果content不为空
            yield content  # 生成content

def stream_chain(topic: str) -> Iterator[str]:
    prompt_template = "Tell me a short joke about {topic}"  # 定义prompt模板
    prompt_value = prompt_template.format(topic=topic)  # 定义prompt
    return stream_chat_model([{"role": "user", "content": prompt_value}])  # 生成stream

def stream_with_qianfan() -> None:
    for chunk in stream_chain("ice cream"):  # 逐条处理stream
        print(chunk, end="", flush=True)  # 打印chunk


if __name__ == "__main__":
    stream_with_qianfan()