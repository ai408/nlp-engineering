# 这个文件包含你的自定义操作，可以用来运行自定义的Python代码。
# 看这个指南如何实现这些操作：
# https://rasa.com/docs/rasa/custom-actions
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from logging import getLogger
logger = getLogger(__name__)  # 获取日志


class ActionGPTFallback(Action):  # 继承Action类

    def name(self) -> Text:
        return "action_gpt_fallback"

    def run(self, dispatcher: CollectingDispatcher,  # CollectingDispatcher表示收集分发器
            tracker: Tracker,  # Tracker跟踪器
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:  # Dict[Text, Any]表示文本和任何类型的字典，domain表示域

        # 获取用户输入
        user_input = tracker.latest_message.get("text")

        # 调用第三方接口处理 out_of_scope 的情况
        # 这里只是一个示例，你需要根据实际情况替换成调用你的第三方接口的代码
        response_from_third_party = self.call_third_party_api(user_input)

        # 将第三方接口返回的信息发送给用户
        dispatcher.utter_message(response_from_third_party)

        return []

    def call_third_party_api(self, user_input):
        # 在这里编写调用第三方接口的代码，返回第三方接口的响应信息
        import requests
        import json
        url = "http://127.0.0.1:7861/chat/knowledge_base_chat"
        data = {
            "query": user_input,
            "knowledge_base_name": "samples",
            "top_k": 3,
            "score_threshold": 1,
            "history": [],
            "stream": False,
            "model_name": "Qwen-1_8B-Chat",
            "temperature": 0.7,
            "max_tokens": 0,
            "prompt_name": "default"
        }
        data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=data, headers=headers)
        response = response.json()
        print(response)

        return response["answer"]
