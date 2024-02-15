from typing import List
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

async def acall_chat_model(messages: List[dict]) -> str:
    client = qianfan.ChatCompletion()  # 创建客户端
    response = await client.ado(  # 异步调用
        model="ERNIE-Bot-turbo",
        messages=messages,
    )
    return response.body['result']  # 返回结果

async def ainvoke_chain(topic: str) -> str:
    prompt_template = "Tell me a short joke about {topic}"
    prompt_value = prompt_template.format(topic=topic)
    messages = [{"role": "user", "content": prompt_value}]
    return await acall_chat_model(messages)

async def async_with_qianfan() -> str:
    response = await ainvoke_chain("ice cream")
    print(response)
    return response


if __name__ == "__main__":
    import asyncio
    asyncio.run(async_with_qianfan())