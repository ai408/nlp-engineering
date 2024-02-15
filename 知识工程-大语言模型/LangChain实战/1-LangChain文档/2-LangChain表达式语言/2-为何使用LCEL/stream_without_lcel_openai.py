from typing import Iterator, List
import openai
from langchain_openai import ChatOpenAI
from api_secret_key import OPENAI_API_KEY, OPENAI_API_PROXY


def chat_with_openai() -> ChatOpenAI:
    """
    使用OpenAI的API进行对话
    """
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo")
    return chat

def stream_chat_model(messages: List[dict]) -> Iterator[str]:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 创建客户端
    stream = client.chat.completions.create(  # 创建stream
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    for response in stream:  # 逐条处理stream
        content = response.choices[0].delta.content  # 获取content
        if content is not None:  # 如果content不为空
            yield content  # 生成content

def stream_chain(topic: str) -> Iterator[str]:
    prompt_template = "Tell me a short joke about {topic}"  # 定义prompt模板
    prompt_value = prompt_template.format(topic=topic)  # 定义prompt
    return stream_chat_model([{"role": "user", "content": prompt_value}])  # 生成stream

def stream_with_openai() -> None:
    for chunk in stream_chain("ice cream"):  # 逐条处理stream
        print(chunk, end="", flush=True)  # 打印chunk


if __name__ == "__main__":
    stream_with_openai()