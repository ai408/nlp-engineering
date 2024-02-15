from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from api_secret_key import OPENAI_API_KEY, OPENAI_API_PROXY


def chat_with_openai() -> ChatOpenAI:
    """
    使用OpenAI的API进行对话
    """
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo")
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
    llm_chat = chat_with_openai()
    asyncio.run(async_lcel(llm_chat))