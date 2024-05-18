import os
import json
import numpy as np

utt_list = []

temp_json_dir = "/home/t-zeqianju/yuancwang/Amphion/temp_jsons/gpt_env_examples"
beach_sum = 0
child_sum = 0
driving_sum = 0
raining_sum = 0
for file in os.listdir(temp_json_dir):
    if file.endswith(".json"):
        with open(os.path.join(temp_json_dir, file), "r") as f:
            temp_json = json.load(f)
    for utt_info in temp_json:
        utt_list.append(utt_info)
        if file.startswith("beach"):
            beach_sum += 1
        elif file.startswith("child"):
            child_sum += 1
        elif file.startswith("driving"):
            driving_sum += 1
        elif file.startswith("raining"):
            raining_sum += 1
print(beach_sum, child_sum, driving_sum, raining_sum)

uid2text_file = "/home/t-zeqianju/yuancwang/Amphion/temp_meta_info/uid2text.json"
with open(uid2text_file, "r") as f:
    uid2text = json.load(f)
text2uid = {v: k for k, v in uid2text.items()}
print(len(text2uid.keys()))

new_utt_list = []

already_used_text = []

for utt_info in utt_list:
    if utt_info["text"] in already_used_text:
        continue
    text = utt_info["text"]
    already_used_text.append(text)
    uid = text2uid[text]
    for i in range(5):
        new_uid = uid + "_" + str(i)
        new_utt_info = utt_info.copy()
        new_utt_info["uid"] = new_uid
        new_utt_list.append(new_utt_info)

# print(new_utt_list)
print(len(new_utt_list))

# Save new_utt_list
new_json_file = "/home/t-zeqianju/yuancwang/Amphion/temp_meta_info/test_temp.json"
with open(new_json_file, "w") as f:
    # it has \u00e9 in the text, so need to use ensure_ascii=False
    json.dump(new_utt_list, f, indent=4, ensure_ascii=False)

