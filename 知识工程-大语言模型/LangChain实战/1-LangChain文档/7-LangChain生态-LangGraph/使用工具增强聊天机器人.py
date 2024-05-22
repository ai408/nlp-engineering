from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from api_secret_key import OPENAI_API_KEY, BASE_URL


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)  # 创建一个状态图
tool = TavilySearchResults(max_results=2)  # 创建一个工具
tools = [tool]  # 将工具放入列表
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_api_base=BASE_URL, model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)  # 将工具绑定到语言模型上


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)  # 添加一个节点
tool_node = ToolNode(tools=[tool])  # 创建一个工具节点
graph_builder.add_node("tools", tool_node)  # 添加一个工具节点
graph_builder.add_conditional_edges(
    "chatbot",  # 起始节点，即聊天机器人节点
    tools_condition,  # 条件函数，即预构建的 tools_condition
)
# 任何时候调用工具，我们都会返回到聊天机器人来决定下一步
graph_builder.add_edge("tools", "chatbot")  # 添加边，即当工具节点满足条件时，返回到聊天机器人节点
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)