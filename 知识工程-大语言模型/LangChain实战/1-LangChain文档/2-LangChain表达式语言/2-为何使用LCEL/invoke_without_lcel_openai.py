from typing import List
import openai
from langchain_openai import ChatOpenAI
from api_secret_key import OPENAI_API_KEY, OPENAI_API_PROXY


def chat_with_openai() -> ChatOpenAI:
    """
    使用OpenAI的API进行对话
    """
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo")
    return chat

def call_chat_model(messages: List[dict]) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 创建客户端
    response = client.chat.completions.create(  # 调用模型
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content  # 返回结果

def invoke_chain(topic: str) -> str:
    prompt_template = "Tell me a short joke about {topic}"
    prompt_value = prompt_template.format(topic=topic)  # 定义prompt
    messages = [{"role": "user", "content": prompt_value}]  # 定义消息
    return call_chat_model(messages)  # 调用模型

def invoke_with_openai() -> str:
    response = invoke_chain("ice cream")
    print(response)
    return response


if __name__ == "__main__":
    invoke_with_openai()