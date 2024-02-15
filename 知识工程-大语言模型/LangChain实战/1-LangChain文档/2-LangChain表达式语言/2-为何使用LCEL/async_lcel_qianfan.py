from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

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

async def async_lcel(model: ChatOpenAI) -> str:
    prompt = ChatPromptTemplate.from_template(  # 定义prompt
        "Tell me a short joke about {topic}"
    )
    output_parser = StrOutputParser()  # 定义输出解析器
    chain = (  # 定义链
        {"topic": RunnablePassthrough()}  # 定义参数
        | prompt
        | model
        | output_parser
    )

    chain_response = await chain.ainvoke("ice cream")  # 异步调用链
    print(chain_response)

    return chain_response


if __name__ == "__main__":
    import asyncio
    llm_chat = chat_with_qianfan()
    asyncio.run(async_lcel(llm_chat))