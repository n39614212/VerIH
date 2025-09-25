import os
import litellm
import traceback
from anthropic import Anthropic
from openai import OpenAI


class OpenAIInterface(object):
    total_cost = 0
    def __init__(self, model="gpt-4o", temperature=0.9, osu_proxy=False):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens=4096
        if osu_proxy:
            self.completion_func = self._osu_proxy
        else:
            self.completion_func = self._offical_api

    def generate(self, system_prompt, user_prompt, formater=None, default_value=None):
        prompt_input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._generate(prompt_input, formater, default_value)

    def _generate(self, prompt_input, formater, default_value):
        # Retry 3 times if LLM error or Format error
        for i in range(3):
            msg = None
            try:
                response = self.completion_func(prompt_input)
                cost = litellm.completion_cost(completion_response=response)
                self.__class__.total_cost += cost

                # Results
                msg = self._extract_msg(response)
                if formater: return formater(msg)
                else: return msg
            except Exception as e:
                traceback.print_exc()
                print(msg)
        return default_value

    def _extract_msg(self, response):
        return response.choices[0].message.content

    def _osu_proxy(self, prompt_input):
        # Check model name
        # curl -X GET https://litellmproxy.osu-ai.org/models -H "Authorization: Bearer $KEY"
        client = OpenAI(
            base_url="https://litellmproxy.osu-ai.org",
            api_key=os.getenv("OSU_API_KEY")
        )

        response = client.chat.completions.create(
            max_tokens=self.max_new_tokens,
            model=self.model,
            messages = prompt_input
        )
        return response

    def _offical_api(self, prompt_input):
        response = litellm.completion(
            model=self.model,
            messages=prompt_input,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return response


import json
import re
def formate_json(msg):
    # Filter outer text by regex
    pattern = r"```(.*?)```"
    match = re.search(pattern, msg, re.DOTALL)
    if match:
        msg = match.group(1)

    # Construct json object
    msg = msg.strip().removeprefix("json").strip()
    res = json.loads(msg)
    return res
