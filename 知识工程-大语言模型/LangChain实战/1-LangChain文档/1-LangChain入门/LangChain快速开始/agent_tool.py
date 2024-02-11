# encoding: utf-8
"""
LangChain快速开始：https://z0yrmerhgi8.feishu.cn/wiki/UwKIwGYcliytiekTKzPco3jnndh
"""

from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub
from langchain.agents import AgentExecutor

from api_secret_key import API_Key, API_SECRET, TAVILY_API_KEY
import os
os.environ["QIANFAN_AK"] = API_Key
os.environ["QIANFAN_SK"] = API_SECRET


# 使用文心一言模型
def chat_with_qianfan() -> QianfanChatEndpoint:
    """文心一言模型"""
    chat = QianfanChatEndpoint(streaming=True,  # 是否使用流式模式
                               model="ERNIE-Bot-turbo",  # 使用的模型
                               **{"top_p": 0.8, "temperature": 0.95, "penalty_score": 1}  # 模型参数
    )
    return chat


def agent_tool(llm: QianfanChatEndpoint):
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")  # 从网页加载文档
    docs = loader.load()  # 加载文档
    embeddings = QianfanEmbeddingsEndpoint()
    text_splitter = RecursiveCharacterTextSplitter(  # 使用递归字符拆分器
        chunk_size=1000,        # 拆分大小
        chunk_overlap=20,      # 重叠大小
        length_function=len,   # 长度函数（默认）
        add_start_index=False, # 添加开始索引（默认）
    )
    documents = text_splitter.split_documents(docs)  # 拆分文档
    vector = FAISS.from_documents(documents, embeddings)  # 创建向量存储
    retriever = vector.as_retriever()  # 创建检索器

    retriever_tool = create_retriever_tool(  # 创建检索工具
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )

    # 设置Tavily搜索工具，这需要一个API密钥
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()  # 创建Tavily搜索工具
    tools = [retriever_tool, search]  # 创建一个包含所有工具的列表

    prompt = hub.pull("hwchase17/openai-functions-agent")
    print(prompt)
    from qianfan_functions_agent import create_qianfan_functions_agent
    agent = create_qianfan_functions_agent(llm, tools, prompt)  # 创建代理
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # 创建代理执行器

    # 问题1：调用代理并询问"langsmith如何帮助测试？"然后打印出答案
    response = agent_executor.invoke({"input": "how can langsmith help with testing?"})
    print(response)

    # 问题2：调用代理执行器来询问"SF（旧金山）的天气怎么样？"
    response = agent_executor.invoke({"input": "what is the weather in SF?"})
    print(response)

    # 问题3：使用代理执行器进行对话，传入聊天历史和用户输入，以便代理可以回答关于"LangSmith如何帮助测试我的LLM应用？"的后续问题
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response = agent_executor.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(response)


if __name__ == "__main__":
    llm = chat_with_qianfan()
    agent_tool(llm)