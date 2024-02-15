from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAI

from api_secret_key import OPENAI_API_KEY, OPENAI_API_PROXY


def llm_with_openai() -> OpenAI:
    """
    OpenAI：用于处理OpenAI的自动完成功能
    ChatOpenAI：用于处理OpenAI的对话功能
    """
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo-instruct")
    return llm

def llm_lcel(llm: OpenAI) -> str:
    prompt = ChatPromptTemplate.from_template(  # 定义prompt
        "Tell me a short joke about {topic}"
    )
    output_parser = StrOutputParser()  # 定义输出解析器
    llm_chain = (  # 定义链
        {"topic": RunnablePassthrough()}  # 定义参数
        | prompt
        | llm
        | output_parser
    )

    chain_response = llm_chain.invoke("ice cream")  # 调用链
    print(chain_response)

    return chain_response


if __name__ == "__main__":
    llm = llm_with_openai()
    llm_lcel(llm)