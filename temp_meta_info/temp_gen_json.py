import os
import json
import numpy as np

utt_list = []

temp_json_dir = "/home/t-zeqianju/yuancwang/Amphion/temp_jsons/gpt_env_examples"
for file in os.listdir(temp_json_dir):
    if file.endswith(".json"):
        with open(os.path.join(temp_json_dir, file), "r") as f:
            temp_json = json.load(f)
    for utt_info in temp_json:
        utt_list.append(utt_info)

uid2text_file = "/home/t-zeqianju/yuancwang/Amphion/temp_meta_info/uid2text.json"
with open(uid2text_file, "r") as f:
    uid2text = json.load(f)
text2uid = {v: k for k, v in uid2text.items()}

new_utt_list = []

for utt_info in utt_list:
    text = utt_info["text"]
    uid = text2uid[text]
    for i in range(5):
        new_uid = uid + "_" + str(i)
        new_utt_info = utt_info.copy()
        new_utt_info["uid"] = new_uid
        new_utt_list.append(new_utt_info)

print(new_utt_list)
print(len(new_utt_list))

# Save new_utt_list
new_json_file = "/home/t-zeqianju/yuancwang/Amphion/temp_meta_info/test_temp.json"
with open(new_json_file, "w") as f:
    # it has \u00e9 in the text, so need to use ensure_ascii=False
    json.dump(new_utt_list, f, indent=4, ensure_ascii=False)

