import pickle
import json

error_path = {}

language_list = {'zh': 0.1, 'en': 0.1, 'ja': 0.1, 'ko': 0.1, 'fr': 0.1, 'de': 0.1}
meta_1k = pickle.load(open('json_path2meta_1k.pkl', "rb"))
meta_5k = pickle.load(open('json_path2meta_5k.pkl', "rb"))
meta_10k = pickle.load(open('json_path2meta_10k.pkl', "rb"))
# print(json_path2meta)

set_1k = set()
for keys, value in meta_1k.items():
    for v in value:
        if v['language'] not in list(language_list.keys()):
            set_1k.add(keys)
error_path['1k'] = list(set_1k)

set_5k = set()
for keys, value in meta_5k.items():
    for v in value:
        if v['language'] not in list(language_list.keys()):
            set_5k.add(keys)
error_path['5k'] = list(set_5k)

set_10k = set()
for keys, value in meta_10k.items():
    for v in value:
        if v['language'] not in list(language_list.keys()):
            set_1k.add(keys)
error_path['10k'] = list(set_10k)

with open("error_path.json", "w") as file:
    json.dump(error_path, file)
