from vllm import LLM, SamplingParams

# 定义批量数据
prompts = [
    "宪法规定的公民法律义务有",
    "属于专门人民法院的是",
    "无效婚姻的种类包括",
    "刑事案件定义",
    "税收法律制度",
]
sampling_params = SamplingParams(temperature=0.1, top_p=0.5, max_tokens=4096)
path = '/data/ssw/llm_model/chatglm3-6b'
llm = LLM(model=path, trust_remote_code=True, tokenizer_mode="auto", tensor_parallel_size=2, dtype="auto")
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")