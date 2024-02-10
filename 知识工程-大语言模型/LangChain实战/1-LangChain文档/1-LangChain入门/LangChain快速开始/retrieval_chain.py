# encoding: utf-8
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


def retrieval_chain_qianfang(llm: QianfanChatEndpoint) -> str:
    # 安装BeautifulSoup库
    # pip install beautifulsoup4
    # 从langchain_community导入文档加载器WebBaseLoader，并加载文档
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")  # 从网页加载文档
    docs = loader.load()  # 加载文档

    # 安装并使用BaichuanTextEmbeddings嵌入模型
    from langchain_community.embeddings import QianfanEmbeddingsEndpoint
    embeddings = QianfanEmbeddingsEndpoint()

    # 安装FAISS库并创建本地向量存储，然后将文档拆分并索引到向量存储中
    # pip install faiss-cpu

    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(  # 使用递归字符拆分器
        chunk_size=1000,        # 拆分大小
        chunk_overlap=20,      # 重叠大小
        length_function=len,   # 长度函数（默认）
        add_start_index=False, # 添加开始索引（默认）
    )
    documents = text_splitter.split_documents(docs)  # 拆分文档
    vector = FAISS.from_documents(documents, embeddings)  # 创建向量存储


    # 方式1：设置链条，结合提供的上下文生成答案
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)  # 创建链条
    # 直接运行链条，传入文档作为上下文
    from langchain_core.documents import Document
    document_chain_response = document_chain.invoke({
        "input": "how can langsmith help with testing?",  # 问题
        "context": [Document(page_content="langsmith can let you visualize test results")]  # 上下文
    })
    print(document_chain_response)


    # 方式2：创建检索链条，用于检索与问题最相关的文档
    from langchain.chains import create_retrieval_chain
    retriever = vector.as_retriever()  # 创建检索器
    retrieval_chain = create_retrieval_chain(retriever, document_chain)  # 创建检索链

    # 调用检索链，查询"langsmith如何帮助测试？"并打印答案
    retrieval_chain_response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})  # 调用检索链
    print(retrieval_chain_response["answer"])
    return retrieval_chain_response["answer"]


if __name__ == '__main__':
    llm = chat_with_qianfan()
    retrieval_chain_response = retrieval_chain_qianfang(llm)
    # print(retrieval_chain_response)