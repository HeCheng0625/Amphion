# [{"wav_path": ...,
# "transcript":...,
# "gender":...,
# "class": background,
# "background": ...,
# "target": [{"reply": ...}]}]

# background type: ["children's voice", "driving or traffic", "raining and thundering", "at the sea beach"]

from openai import OpenAI
from tqdm import tqdm
import json
import ast
import os


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


client = OpenAI(api_key="")
completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {
            "role": "system",
            "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        },
        {
            "role": "user",
            "content": "## Instruction:\n"
            "You are tasked with simulating a conversation between a simulated speaker and yourself, ChatGPT. \n"
            "Could you provide examples of how a speech dialogue model would adapt its responses based on different environmental sounds? Please include scenarios where the model detects distinctive background noises (such as in a busy marketplace, at a beach, children's vocice, or during a thunderstorm). For each scenario, provide a brief description of the background sound and a suitable model response that aligns with the context. \n"
            "Now, let's firstly fcous on 4 different background types: children's voice, driving or traffic, raining and thundering, at the sea beach. \n"
            'Your response must be formatted in JSON, each entry should contain the following keys: "text", "background", "reply" \n'
            "I will give you some examples: \n"
            '{text: "Can you tell me about the effects of adult beverages on health?", background: "children\'s voice", reply: ["It sounds like there might be little ears around! If you\'re asking for general health information, it\'s important to maintain a balanced lifestyle, which includes moderation in all things."]}\n'
            '{text: "What should I wear today?", background: "raining and thundering", reply: ["Given the rainy and windy weather outside, I recommend wearing a waterproof jacket and sturdy, water-resistant shoes. Would you like a reminder to bring an umbrella?"]}\n'
            "OK, now it's your turn! Please provide examples of how a speech dialogue model would adapt its responses based on different environmental sounds, let's firstly fcous on 4 different background types: children's voice, driving or traffic, raining and thundering, at the sea beach, and in my examples, each \"reply\" contains only one response, you should provide 5 proper responses. \n"
            'For each type of the 4 different environmental sounds, please generate 5 examples, which means total 20 dict entries {"text": ..., "background":..., "reply":[...]} in the JSON format. \n',
        },
    ],
)


# print(completion.choices)
response = completion.choices[0].message.content
print(response)
# save the response to a file
with open("temp_jsons/gpt_env_examples_txt/2.txt", "w") as f:
    f.write(response)

response_json = completion.choices[0].message.content.lstrip("```json\n").strip("\n```")
response_json = ast.literal_eval(response_json)
# save the response to a file
with open("temp_jsons/gpt_env_examples/2.json", "w") as f:
    json.dump(response_json, f, indent=4)
