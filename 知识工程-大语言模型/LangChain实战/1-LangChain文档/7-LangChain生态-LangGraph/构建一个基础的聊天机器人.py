from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from api_secret_key import OPENAI_API_KEY, BASE_URL


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)  # 创建一个状态图形结构
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_api_base=BASE_URL, model="gpt-3.5-turbo", temperature=0)
graph_builder.add_node("chatbot", chatbot)  # 第一个参数表示节点的唯一名称，第二个参数表示每次使用节点时将调用的函数或对象
graph_builder.set_entry_point("chatbot")  # 设置入口节点
graph_builder.set_finish_point("chatbot")  # 设置结束节点
graph = graph_builder.compile()  # 编译图形结构

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)