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


for index in tqdm(range(7, 10)):

    client = OpenAI(api_key="")
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are ChatGPT 4, a large language model trained by OpenAI. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            },
            {
                "role": "user",
                "content": "## Instruction:\n"
                "You are tasked with simulating a conversation between a simulated speaker and yourself, ChatGPT. \n"
                "Could you provide examples of how a speech dialogue model would adapt its responses based on different environmental sounds? Please include scenarios where the model detects distinctive background noises (such as in a busy marketplace, at a beach, children's vocice, or during a thunderstorm). For each scenario, provide a brief description of the background sound and a suitable model response that aligns with the context. \n"
                "Now, let's firstly fcous on a special background noise types: raining or thundering. \n"
                'Your response must be formatted in JSON, each entry should contain the four keys: "text", "background", "reply", and "reason", "reason" should fcous on why the reply may change due to the presence of children\'s voices. \n'
                "I will give you some examples: \n"
                '{text: "Can you tell me about the effects of adult beverages on health?", background: "children\'s voice", reply: ["It sounds like there might be little ears around! If you\'re asking for general health information, it\'s important to maintain a balanced lifestyle, which includes moderation in all things."], reason: "The large model can answer the question about the effects of adult beverages, but because there are children\'s voices in the background, it is not suitable for direct answers, but for discussing more general health topics."}\n'
                '{text: "What should I wear today?", background: "raining or thundering", reply: ["Given the rainy and windy weather outside, I recommend wearing a waterproof jacket and sturdy, water-resistant shoes. Would you like a reminder to bring an umbrella?"], reason: "Since there are sounds of rain and thunder in the background, it is recommended to bring an umbrella."}\n'
                '{text: "Where\'s a good place to eat around here?", background: "driving or traffic", reply: ["Since you\'re on the road, I\'ll find something easy to access. There\'s a highly-rated grill just off the next exit. Want me to guide you there?"], reason: "..."}\n'
                "OK, now it's your turn! Please provide examples of how a speech dialogue model would adapt its responses based on different environmental sounds, let's firstly fcous on a special background types: raining and thundering.\n"
                "Again, the scene we are focusing on is an adult conversation but the reply may change due to the presence of raining or thundering. \n"
                'And in my examples, each "reply" contains only one response, you should provide 5 proper responses for each example. \n'
                "Please generate 5 examples, each example contains 5 responses, and for each example, only give one reason for general. \n",
            },
        ],
    )

    # print(completion.choices)
    response = completion.choices[0].message.content
    print(response)
    # save the response to a file
    with open(
        "temp_jsons/gpt_env_examples_txt/raining_{}.txt".format(str(index)), "w"
    ) as f:
        f.write(response)

    response_json = (
        completion.choices[0].message.content.lstrip("```json\n").strip("\n```")
    )
    response_json = ast.literal_eval(response_json)
    # save the response to a file
    with open(
        "temp_jsons/gpt_env_examples/raining_{}.json".format(str(index)), "w"
    ) as f:
        json.dump(response_json, f, indent=4)
