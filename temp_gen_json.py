import os
import json
import librosa
from tqdm import tqdm

base_path = "/home/t-zeqianju/yuancwang/temp_test_dataset/libritts-train-clean-100"
wav_dir_path = "train-clean-100"

prompt_json = []

for spk in tqdm(os.listdir(os.path.join(base_path, wav_dir_path))):
    for chapter in os.listdir(os.path.join(base_path, wav_dir_path, spk)):
        for file in os.listdir(os.path.join(base_path, wav_dir_path, spk, chapter)):
            if file.endswith(".wav"):
                prompt_info = {}
                file_path = os.path.join(wav_dir_path, spk, chapter, file)
                wav_len = librosa.get_duration(
                    filename=os.path.join(base_path, wav_dir_path, spk, chapter, file)
                )
                text_file = os.path.join(
                    base_path, wav_dir_path, spk, chapter, file
                ).replace(".wav", ".normalized.txt")
                text = open(text_file).read()
                # print(text)
                prompt_info["path"] = file_path
                prompt_info["text"] = text
                prompt_info["duration"] = wav_len
                if wav_len >= 3 and wav_len <= 8:
                    prompt_json.append(prompt_info)

with open("temp_meta_info/libritts_train_clean_100.json", "w") as f:
    json.dump(prompt_json, f, indent=4)
