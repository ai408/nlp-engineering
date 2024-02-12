# encoding: utf-8
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import QianfanChatEndpoint

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


def conversation_retrieval_chain_qianfan(llm: QianfanChatEndpoint) -> str:
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.embeddings import QianfanEmbeddingsEndpoint
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # 方式1：create_history_aware_retriever：创建一个链，它接受对话历史并返回文档
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")  # 从网页加载文档
    docs = loader.load()  # 加载文档
    embeddings = QianfanEmbeddingsEndpoint()
    text_splitter = RecursiveCharacterTextSplitter(  # 使用递归字符拆分器
        chunk_size=500,        # 拆分大小
        chunk_overlap=20,      # 重叠大小
        length_function=len,   # 长度函数（默认）
        add_start_index=False, # 添加开始索引（默认）
    )
    documents = text_splitter.split_documents(docs)  # 拆分文档
    vector = FAISS.from_documents(documents, embeddings)  # 创建向量存储
    retriever = vector.as_retriever()  # 创建检索器

    prompt = ChatPromptTemplate.from_messages([  # 从消息创建一个模板
        MessagesPlaceholder(variable_name="chat_history"),  # 消息占位符
        ("user", "{input}"),  # 用户输入
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    # 通过传入一个实例来测试用户提出的后续问题。这个实例中用户询问LangSmith是否可以帮助测试他们的LLM应用程序
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    retriever_chain_response1 = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    # 对话流程：用户（LangSmith能够帮助测试我的LLM应用程序吗？）->机器（是的！）->用户（告诉我怎么做）->机器（XXX）
    print("retriever_chain_response1", retriever_chain_response1)


    # 方式2：创建一个新的链来考虑检索到的文档，继续对话
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),  # 系统消息，回答用户的问题基于下面的上下文
        MessagesPlaceholder(variable_name="chat_history"),  # 消息占位符
        ("user", "{input}"),  # 用户输入
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    # 创建最终的检索链，将检索链和文档链结合
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    # 端到端测试这个链，可以看到这提供了一个连贯的答案 - 已经成功地将检索链变成了一个聊天机器人
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    retrieval_chain_response2 = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })

    # 对话流程：用户（告诉我怎么做）[context+chat_history]->机器（XXX）
    print('retrieval_chain_response2["answer"]', retrieval_chain_response2["answer"])

    return retrieval_chain_response2["answer"]


if __name__ == "__main__":
    llm = chat_with_qianfan()
    conversation_retrieval_chain_response = conversation_retrieval_chain_qianfan(llm)
    # print("conversation_retrieval_chain_response", conversation_retrieval_chain_response)