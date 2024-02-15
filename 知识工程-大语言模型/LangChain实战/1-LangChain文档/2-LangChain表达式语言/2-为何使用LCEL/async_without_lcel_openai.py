from typing import List
import openai
from langchain_openai import ChatOpenAI
from api_secret_key import OPENAI_API_KEY, OPENAI_API_PROXY


def chat_with_openai() -> ChatOpenAI:
    """
    使用OpenAI的API进行对话
    """
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_API_PROXY, model="gpt-3.5-turbo")
    return chat

async def acall_chat_model(messages: List[dict]) -> str:
    async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content

async def ainvoke_chain(topic: str) -> str:
    prompt_template = "Tell me a short joke about {topic}"
    prompt_value = prompt_template.format(topic=topic)
    messages = [{"role": "user", "content": prompt_value}]
    return await acall_chat_model(messages)

async def async_with_openai() -> str:
    response = await ainvoke_chain("ice cream")
    print(response)
    return response


if __name__ == "__main__":
    import asyncio
    asyncio.run(async_with_openai())