# [{"wav_path": ...,
# "transcript":...,
# "gender":...,
# "class": background,
# "background": ...,
# "target": [{"reply": ...}]}]

# background: ["children speaking, playing", "driving or traffic", "raining or thundering", "sea beach", ...]

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
