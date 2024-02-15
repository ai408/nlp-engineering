from langchain_community.chat_models import ChatAnthropic
import anthropic


def chat_with_anthropic() -> ChatAnthropic:
    chat = ChatAnthropic(model="claude-2", anthropic_api_key="my-api-key")  # 定义anthropic
    return chat

def call_anthropic(prompt_value: str) -> str:
    anthropic_client = anthropic.Anthropic(api_key="my-api-key")  # 创建客户端
    response = anthropic_client.completions.create(
        model="claude-2",
        prompt=prompt_value,
        max_tokens_to_sample=256,
    )
    return response.completion

def invoke_anthropic_chain(topic: str) -> str:
    prompt_template = "Tell me a short joke about {topic}"
    anthropic_template = f"Human:\n\n{prompt_template}\n\nAssistant:"
    prompt_value = anthropic_template.format(topic=topic)
    return call_anthropic(prompt_value)

def privider_lcel():
    invoke_anthropic_chain("ice cream")


if __name__ == "__main__":
    chat = chat_with_anthropic()
    privider_lcel()