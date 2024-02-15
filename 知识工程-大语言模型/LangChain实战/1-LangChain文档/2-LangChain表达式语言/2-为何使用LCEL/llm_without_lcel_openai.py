import openai
from langchain_openai import OpenAI
from api_secret_key import OPENAI_API_KEY, OPENAI_API_PROXY


def llm_with_openai() -> OpenAI:
    """
    OpenAI：用于处理OpenAI的自动完成功能
    ChatOpenAI：用于处理OpenAI的对话功能
    """
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo-instruct")
    return llm

def call_llm(prompt_value: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 创建客户端
    response = client.completions.create(  # 调用模型
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_value,
    )
    return response.choices[0].text  # 返回结果

def invoke_llm_chain(topic: str) -> str:
    prompt_template = "Tell me a short joke about {topic}"
    prompt_value = prompt_template.format(topic=topic)  # 定义prompt
    return call_llm(prompt_value)  # 调用模型

def llm_with_openai() -> str:
    response = invoke_llm_chain("ice cream")
    print(response)
    return response


if __name__ == "__main__":
    llm_with_openai()