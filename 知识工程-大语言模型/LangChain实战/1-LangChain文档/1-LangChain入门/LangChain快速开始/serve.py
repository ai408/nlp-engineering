# encoding: utf-8
from typing import List

from fastapi import FastAPI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

from qianfan_functions_agent import create_qianfan_functions_agent
from api_secret_key import API_Key, API_SECRET, TAVILY_API_KEY
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


def langserve(llm: QianfanChatEndpoint) -> AgentExecutor:
    # 1.加载检索器
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(  # 使用递归字符拆分器
        chunk_size=1000,  # 拆分大小
        chunk_overlap=20,  # 重叠大小
        length_function=len,  # 长度函数（默认）
        add_start_index=False,  # 添加开始索引（默认）
    )
    documents = text_splitter.split_documents(docs)
    embeddings = QianfanEmbeddingsEndpoint()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    # 2. 创建工具
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults()
    tools = [retriever_tool, search]


    # 3.创建智能体
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # agent = create_openai_functions_agent(llm, tools, prompt)
    agent = create_qianfan_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


# 4.App定义
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)


# 5.增加链路由
# 我们需要添加这些输入/输出模式，因为当前的AgentExecutor缺乏模式。
class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )

class Output(BaseModel):
    output: str

llm = chat_with_qianfan()  # 初始化文心一言模型
agent_executor = langserve(llm)  # 初始化智能体
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)