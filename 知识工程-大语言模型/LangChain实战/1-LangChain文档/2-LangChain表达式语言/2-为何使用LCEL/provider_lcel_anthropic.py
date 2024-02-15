from langchain_community.chat_models import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def chat_with_anthropic() -> ChatAnthropic:
    chat = ChatAnthropic(model="claude-2", anthropic_api_key="my-api-key")  # 定义anthropic
    return chat

def privider_lcel(anthropic: ChatAnthropic) -> str:
    prompt = ChatPromptTemplate.from_template(  # 定义prompt
        "Tell me a short joke about {topic}"
    )
    output_parser = StrOutputParser()  # 定义输出解析器

    anthropic_chain = (  # 定义anthropic_chain
            {"topic": RunnablePassthrough()}
            | prompt
            | anthropic
            | output_parser
    )
    chain_response = anthropic_chain.invoke("ice cream")
    print(chain_response)

    return chain_response


if __name__ == "__main__":
    chat = chat_with_anthropic()
    privider_lcel(chat)