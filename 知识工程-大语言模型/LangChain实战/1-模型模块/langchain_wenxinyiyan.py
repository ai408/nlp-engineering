from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_wenxin import ChatWenxin
import os
from api_secret_key import API_Key, API_SECRET


def chat_with_openai():
    """
    使用OpenAI的API进行对话
    """
    os.environ["OPENAI_API_KEY"] = ""
    chat = ChatOpenAI(temperature=0)
    response = chat.predict_messages([  # 生成回复
        HumanMessage(
            content = (
                "Translate this sentence from English to Chinese. "
                "I love programming."
            )
        )
    ])
    print(response)  # 打印回复


def chat_with_wenxin():
    """
    使用文心一言接口进行对话
    """
    # WENXIN_APP_Key = ""  # 设置API Key
    # WENXIN_APP_SECRET = ""  # 设置Secret Key
    WENXIN_APP_Key = API_Key
    WENXIN_APP_SECRET = API_SECRET

    chat_model = ChatWenxin(
        temperature = 0.9,  # 表示生成多样性，越大生成的句子越多样，但是也越不准确
        model = "ernie-bot-turbo",  # 表示使用的模型
        baidu_api_key = WENXIN_APP_Key,  # API Key
        baidu_secret_key = WENXIN_APP_SECRET,  # Secret Key
        verbose=True,  # 表示是否打印调试信息
    )

    response_chat = chat_model([ HumanMessage(content="你是谁呢？") ])
    print(response_chat)


def chat_with_qianfan():
    """For basic init and call"""
    import os
    from langchain_community.chat_models import QianfanChatEndpoint
    from langchain_core.language_models.chat_models import HumanMessage

    # os.environ["QIANFAN_AK"] = ""
    # os.environ["QIANFAN_SK"] = ""
    os.environ["QIANFAN_AK"] = API_Key
    os.environ["QIANFAN_SK"] = API_SECRET

    chat = QianfanChatEndpoint(streaming=True)
    messages = [HumanMessage(content="Hello")]

    # 基本用法
    result = chat.invoke(messages)  # 同步调用
    print(result)  # 打印回复
    # await chat.ainvoke(messages)  # 异步调用
    # chat.batch([messages])  # 批量调用


    # 流式调用
    try:
        for chunk in chat.stream(messages):
            print(chunk.content, end="", flush=True)
    except TypeError as e:
        print("")


    # 在千帆中使用不同的模型
    chatBot = QianfanChatEndpoint(
        streaming=True,
        model="ERNIE-Bot",
    )
    messages = [HumanMessage(content="Hello")]
    result = chatBot.invoke(messages)
    print(result)


    # 模型参数，仅仅ERNIE-Bot和ERNIE-Bot-turbo支持temperature、top_p、penalty_score
    result = chat.invoke(
        [HumanMessage(content="Hello")],
        **{"top_p": 0.4, "temperature": 0.1, "penalty_score": 1},
    )
    print(result)


if __name__ == '__main__':
    # chat_with_openai()

    # chat_with_wenxin()

    chat_with_qianfan()