import qianfan
from langchain_openai import ChatOpenAI
from api_secret_key import API_Key, API_SECRET
import os
os.environ["QIANFAN_AK"] = API_Key
os.environ["QIANFAN_SK"] = API_SECRET


def chat_with_openai() -> ChatOpenAI:
    """
    使用OpenAI的API进行对话
    """
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo")
    return chat

def call_llm(prompt_value: str) -> str:
    client = qianfan.Completion()  # 创建客户端
    response = client.do(  # 调用模型
        model="ERNIE-Bot-turbo",
        prompt=prompt_value,
    )
    return response.body['result']  # 返回结果

def invoke_llm_chain(topic: str) -> str:
    prompt_template = "Tell me a short joke about {topic}"
    prompt_value = prompt_template.format(topic=topic)  # 定义prompt
    return call_llm(prompt_value)  # 调用模型

def llm_with_qianfan() -> str:
    response = invoke_llm_chain("ice cream")
    print(response)
    return response


if __name__ == "__main__":
    llm_with_qianfan()