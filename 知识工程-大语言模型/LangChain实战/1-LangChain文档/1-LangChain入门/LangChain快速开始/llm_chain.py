from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import BaseMessage

from api_secret_key import API_Key, API_SECRET


# 使用文心一言模型
def chat_with_qianfan() -> QianfanChatEndpoint:
    """For basic init and call"""
    import os
    from langchain_community.chat_models import QianfanChatEndpoint
    # os.environ["QIANFAN_AK"] = ""
    # os.environ["QIANFAN_SK"] = ""
    os.environ["QIANFAN_AK"] = API_Key
    os.environ["QIANFAN_SK"] = API_SECRET
    chat = QianfanChatEndpoint(streaming=True,  # 是否使用流式模式
                               model="ERNIE-Bot-turbo",  # 使用的模型
                               **{"top_p": 0.8, "temperature": 0.95, "penalty_score": 1}  # 模型参数
                               )
    return chat


def llm_chain(llm: QianfanChatEndpoint) -> BaseMessage:
    # 调用LLM询问关于LangSmith的问题
    result = llm.invoke("LangSmith如何帮助测试？")
    # print("llm_chain:", result)
    return result


def llm_prompt_llm_chain(llm: QianfanChatEndpoint) -> BaseMessage:
    # 使用用户和系统消息设置提示模板
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是世界级的技术文档撰写者。"),
        ("user", "{input}")
    ])

    # 将提示模板与LLM结合创建一个简单的链
    chain = prompt | llm

    # 使用输入调用链
    result = chain.invoke({"input": "LangSmith如何帮助测试？"})
    # print("llm_prompt_llm_chain:", result)
    return result


def llm_prompt_llm_parser_chain(llm: QianfanChatEndpoint) -> str:
    # 使用用户和系统消息设置提示模板
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是世界级的技术文档撰写者。"),
        ("user", "{input}")
    ])

    # 添加一个输出解析器，将聊天消息转换为字符串
    from langchain_core.output_parsers import StrOutputParser
    output_parser = StrOutputParser()

    # 更新链条，加入输出解析器并再次调用
    chain = prompt | llm | output_parser
    result = chain.invoke({"input": "LangSmith如何帮助测试？"})

    # print("llm_prompt_llm_parser_chain:", result)
    return result


if __name__ == "__main__":
    llm = chat_with_qianfan()

    llm_chain_result = llm_chain(llm)
    print("llm_chain:", llm_chain_result)

    llm_prompt_llm_chain_result = llm_prompt_llm_chain(llm)
    print("llm_prompt_llm_chain:", llm_prompt_llm_chain_result)

    llm_prompt_llm_parser_chain_result = llm_prompt_llm_parser_chain(llm)
    print("llm_prompt_llm_parser_chain:", llm_prompt_llm_parser_chain_result)