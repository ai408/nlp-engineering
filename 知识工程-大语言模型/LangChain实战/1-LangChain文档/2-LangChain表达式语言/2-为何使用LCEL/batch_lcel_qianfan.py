from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

def batch_lcel(model: QianfanChatEndpoint) -> None:
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

    chain_response = chain.batch(["ice cream", "spaghetti", "dumplings"])  # 批量运行
    for response in chain_response:  # 打印结果
        print(response)


if __name__ == "__main__":
    llm_chat = chat_with_qianfan()
    batch_lcel(llm_chat)