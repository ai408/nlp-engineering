from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

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


def rag_search_demo(model: QianfanChatEndpoint) -> str:
    # 安装并使用QianfanEmbeddingsEndpoint嵌入模型
    from langchain_community.embeddings import QianfanEmbeddingsEndpoint
    embeddings = QianfanEmbeddingsEndpoint(chunk_size=1000)

    vectorstore = DocArrayInMemorySearch.from_texts(  # 从文本中构建向量检索
        ["harrison worked at kensho", "bears like to eat honey"],  # 文本列表：[harrison在kensho工作，熊喜欢吃蜂蜜]
        embedding=embeddings,  # 使用的嵌入模型
    )
    retriever = vectorstore.as_retriever()  # 将向量检索转换为检索器

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)  # 从模板创建提示
    output_parser = StrOutputParser()  # 创建输出解析器

    setup_and_retrieval = RunnableParallel(  # 并行运行
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser  # 创建运行链

    chain_response = chain.invoke("where did harrison work?")  # 调用运行链：harrison在哪里工作？
    print(chain_response)

    return chain_response


if __name__ == '__main__':
    model = chat_with_qianfan()
    chain_response = rag_search_demo(model)
    print(chain_response)