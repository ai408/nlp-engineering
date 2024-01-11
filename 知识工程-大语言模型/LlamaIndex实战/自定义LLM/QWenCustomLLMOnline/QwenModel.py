from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
app = Flask(__name__)


class QwenModel:
    def __init__(self, pretrained_model_name_or_path):
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, device_map="cpu", trust_remote_code=True)  # CPU方式加载模型
        # self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="cpu", trust_remote_code=True)  # CPU方式加载模型

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, device_map="cuda", trust_remote_code=True)  # GPU方式加载模型
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="cuda", trust_remote_code=True)  # GPU方式加载模型
        self.model = self.model.float()

    def generate_completion(self, prompt):
        # inputs = self.tokenizer.encode(prompt, return_tensors="pt")  # CPU方式加载模型
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").cuda()  # GPU方式加载模型
        outputs = self.model.generate(inputs, max_length=128)
        response = self.tokenizer.decode(outputs[0])
        return response


pretrained_model_name_or_path = r'L:\20230713_HuggingFaceModel\20230925_Qwen\Qwen-1_8B'
qwen_model = QwenModel(pretrained_model_name_or_path)


@app.route('/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data.get('prompt', '')
    result = qwen_model.generate_completion(prompt)
    return jsonify({'text': result})

@app.route('/stream_complete', methods=['POST'])
def stream_complete():
    data = request.get_json()
    prompt = data.get('prompt', '')
    result = list(qwen_model.generate_completion(prompt))
    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=False, port=5050, host='0.0.0.0')