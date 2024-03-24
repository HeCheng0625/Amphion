import os
import librosa
import soundfile as sf

path = "/home/t-zeqianju/yuancwang/AmphionOpen/data/《洪洋洞》张艳栋-国家京剧院.WAV"
y, sr = librosa.load(path, sr=16000)
sf.write("test.wav", y, sr)
