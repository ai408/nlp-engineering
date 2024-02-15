from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from api_secret_key import API_Key, API_SECRET
import os
os.environ["QIANFAN_AK"] = API_Key
os.environ["QIANFAN_SK"] = API_SECRET


# 使用文心一言模型
def llm_with_qianfan() -> QianfanLLMEndpoint:
    """For basic init and call"""
    llm = QianfanLLMEndpoint(streaming=True,  # 是否使用流式模式
                               model="ERNIE-Bot-turbo",  # 使用的模型
                               **{"top_p": 0.8, "temperature": 0.95, "penalty_score": 1}  # 模型参数
    )
    return llm

def llm_lcel(llm: QianfanLLMEndpoint) -> str:
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
    llm = llm_with_qianfan()
    llm_lcel(llm)