from typing import Annotated, Union
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from api_secret_key import OPENAI_API_KEY, BASE_URL, TAVILY_API_KEY


memory = SqliteSaver.from_conn_string(":memory:")
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_api_base=BASE_URL, model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)


# 第1次问题：测试记忆
config = {"configurable": {"thread_id": "1"}}
user_input = "Hi there! My name is Will."
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

# 第2次问题：通过thread_id共享记忆
user_input = "Remember my name?"
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

# 第3次问题：不同thread_id，遗忘记忆
events = graph.stream(
    {"messages": [("user", user_input)]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)  # 获取图的当前状态
print(snapshot)
print(snapshot.next)  # 获取图的下一个状态