from api_secret_key import OPENAI_API_KEY, BASE_URL
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Literal


#例子1：简单的LangChain例子###################################################################################################################
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_api_base=BASE_URL, model="gpt-3.5-turbo", temperature=0)
graph = MessageGraph()  # 用于管理聊天流程的图形结构
graph.add_node("oracle", model)  # 给节点绑定模型
graph.add_edge("oracle", END)  # 给节点绑定结束节点
graph.set_entry_point("oracle")  # 设置入口节点
runnable = graph.compile()  # 编译图形结构

# 可视化图结构
# from IPython.display import Image, display
# try:
#     display(Image(runnable.get_graph(xray=True).draw_mermaid_png()))
# except:
#     # This requires some extra dependencies and is optional
#     pass
print(runnable.invoke(HumanMessage("What is 1 + 1?")))


#例子2：简单的带条件边的LangChain例子###################################################################################################################
# 通过注解方式添加工具函数
@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_api_base=BASE_URL, model="gpt-3.5-turbo", temperature=0)
model_with_tools = model.bind_tools(tools=[multiply])  # 绑定工具函数
graph = MessageGraph()  # 用于管理聊天流程的图形结构
graph.add_node("oracle", model_with_tools)  # 给节点绑定模型
tool_node = ToolNode([multiply])  # 创建工具节点
graph.add_node("multiply", tool_node)  # 给节点绑定工具节点
graph.add_edge("multiply", END)
graph.set_entry_point("oracle")

def router(state: list[BaseMessage]) -> Literal["multiply", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])  # 获取工具函数调用
    if len(tool_calls):
        return "multiply"
    else:
        return END

graph.add_conditional_edges("oracle", router)  # 添加条件边
runnable = graph.compile()
# try:
#     display(Image(runnable.get_graph(xray=True).draw_mermaid_png()))
# except:
#     # This requires some extra dependencies and is optional
#     pass
print(runnable.invoke(HumanMessage("What is 123 * 456?")))
print(runnable.invoke(HumanMessage("What is your name?")))