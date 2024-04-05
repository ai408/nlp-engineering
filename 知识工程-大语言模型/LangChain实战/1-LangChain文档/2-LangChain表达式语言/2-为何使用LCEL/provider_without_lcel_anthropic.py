import anthropic


def call_anthropic(prompt_value: str) -> str:
    anthropic_client = anthropic.Anthropic(api_key="")  # 创建客户端
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
    response = privider_lcel()
    print(response)